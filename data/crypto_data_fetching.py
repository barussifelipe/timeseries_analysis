import math
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


ALTFINS_BASE_URL = 'https://altfins.com/api/v2/public'
HISTORY_DB = r'D:\DBs\timeseries_analysis\history_coverage.db'
CRYPTO_LOG = r'D:\DBs\timeseries_analysis\crypto_history_scan.txt'
CRYPTO_FROM = '2019-01-01T00:00:00Z'
CRYPTO_AS_OF = '2026-01-01T00:00:00Z'
CRYPTO_INTERVALS = {
    'DAILY': 'crypto_daily_history',
    'MINUTES15': 'crypto_intraday_history',
}
OHLCV_FIELDS = ('open', 'high', 'low', 'close', 'volume')
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


def _download_failure_status(error):
    message = error.lower()
    if 'rate limit' in message or '429' in message or 'too many requests' in message:
        return 'rate_limited'
    if 'timeout' in message or 'connection' in message or 'temporarily unavailable' in message:
        return 'transient'
    return 'failed'


def _scan_logger(log_filename):
    logger = logging.getLogger('crypto_history_scan')
    logger.setLevel(logging.INFO)
    absolute_path = str(Path(log_filename).resolve())
    if not any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == absolute_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(absolute_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    return logger


def _create_crypto_tables(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS crypto_symbols (
            symbol TEXT PRIMARY KEY,
            friendly_name TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )
    ''')
    for table in CRYPTO_INTERVALS.values():
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {table} (
                symbol TEXT NOT NULL,
                time TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, time)
            ) WITHOUT ROWID
        ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS crypto_downloads (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_page INTEGER NOT NULL DEFAULT 0,
            total_pages INTEGER,
            from_time TEXT NOT NULL,
            to_time TEXT NOT NULL,
            first_time TEXT,
            last_time TEXT,
            last_error TEXT,
            updated_at TEXT,
            PRIMARY KEY (symbol, interval)
        ) WITHOUT ROWID
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS crypto_daily_volatility (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            realized_variance REAL NOT NULL,
            bars INTEGER NOT NULL,
            PRIMARY KEY (symbol, date)
        ) WITHOUT ROWID
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS crypto_history_coverage (
            symbol TEXT PRIMARY KEY,
            daily_before_na INTEGER NOT NULL,
            daily_after_na INTEGER NOT NULL,
            intraday_days_before_na INTEGER NOT NULL,
            intraday_days_after_na INTEGER NOT NULL,
            aligned_days_after_na INTEGER NOT NULL,
            full_period INTEGER NOT NULL,
            first_date TEXT,
            last_date TEXT
        ) WITHOUT ROWID
    ''')


def _session(api_key=None):
    api_key = api_key or os.environ.get('ALTFINS_API_KEY')
    if not api_key:
        raise ValueError('Set ALTFINS_API_KEY or pass api_key explicitly')
    session = requests.Session()
    session.headers.update({
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'X-API-KEY': api_key,
    })
    return session


def _request_json(session, method, path, *, backoff_seconds=(5, 15, 30), **kwargs):
    global _NEXT_REQUEST_AT
    for attempt in range(len(backoff_seconds) + 1):
        with _REQUEST_LOCK:
            wait = _NEXT_REQUEST_AT - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _NEXT_REQUEST_AT = time.monotonic() + 60 / 140
        response = None
        try:
            response = session.request(
                method, f'{ALTFINS_BASE_URL}/{path}', timeout=60, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            retryable = (
                response is None or response.status_code == 429 or response.status_code >= 500
            )
            if not retryable or attempt == len(backoff_seconds):
                raise
            time.sleep(backoff_seconds[attempt])


def fetch_altfins_symbols(api_key=None, session=None):
    """Return every symbol exposed by altFINS; no instrument filtering."""
    own_session = session is None
    session = session or _session(api_key)
    try:
        payload = _request_json(session, 'GET', 'symbols')
    finally:
        if own_session:
            session.close()

    if isinstance(payload, dict):
        payload = payload.get('content', payload.get('symbols', []))
    symbols = {
        item['name']: item.get('friendlyName') or item['name']
        for item in payload
        if item.get('name')
    }
    return sorted(symbols.items())


def _history_rows(records):
    for record in records:
        yield (
            record['symbol'],
            record['time'],
            *(None if record.get(field) is None else float(record[field])
              for field in OHLCV_FIELDS),
        )


def _download_interval(
    conn,
    session,
    symbol,
    interval,
    from_time,
    to_time,
    page_size,
):
    table = CRYPTO_INTERVALS[interval]
    state = conn.execute(
        '''
        SELECT status, next_page, from_time, to_time
        FROM crypto_downloads
        WHERE symbol = ? AND interval = ?
        ''',
        (symbol, interval),
    ).fetchone()
    if state and state[0] == 'success' and state[2:] == (from_time, to_time):
        return
    if state and state[2:] != (from_time, to_time):
        conn.execute(f'DELETE FROM {table} WHERE symbol = ?', (symbol,))
        conn.execute(
            '''
            UPDATE crypto_downloads
            SET status = 'pending', next_page = 0, total_pages = NULL,
                from_time = ?, to_time = ?
            WHERE symbol = ? AND interval = ?
            ''',
            (from_time, to_time, symbol, interval),
        )
        page = 0
    else:
        page = state[1] if state else 0

    conn.execute(
        '''
        INSERT INTO crypto_downloads
            (symbol, interval, status, attempts, next_page, from_time, to_time)
        VALUES (?, ?, 'running', 1, ?, ?, ?)
        ON CONFLICT(symbol, interval) DO UPDATE SET
            status = 'running', attempts = attempts + 1, last_error = NULL
        ''',
        (symbol, interval, page, from_time, to_time),
    )
    conn.commit()

    while True:
        payload = _request_json(
            session,
            'POST',
            'ohlcv/history-requests',
            params=[('page', page), ('size', page_size), ('sort', 'time,asc')],
            json={
                'symbol': symbol,
                'timeInterval': interval,
                'from': from_time,
                'to': to_time,
            },
        )
        records = payload.get('content', []) if isinstance(payload, dict) else payload
        conn.executemany(
            f'''
            INSERT OR REPLACE INTO {table}
                (symbol, time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            _history_rows(records),
        )
        total_pages = payload.get('totalPages', page + 1) if isinstance(payload, dict) else page + 1
        finished = not records or page + 1 >= total_pages
        conn.execute(
            '''
            UPDATE crypto_downloads
            SET status = ?, next_page = ?, total_pages = ?, updated_at = ?
            WHERE symbol = ? AND interval = ?
            ''',
            (
                'success' if finished else 'running',
                page + 1,
                total_pages,
                datetime.now(timezone.utc).isoformat(),
                symbol,
                interval,
            ),
        )
        conn.commit()
        if finished:
            break
        page += 1

    first_time, last_time = conn.execute(
        f'SELECT MIN(time), MAX(time) FROM {table} WHERE symbol = ?',
        (symbol,),
    ).fetchone()
    conn.execute(
        '''
        UPDATE crypto_downloads
        SET first_time = ?, last_time = ?
        WHERE symbol = ? AND interval = ?
        ''',
        (first_time, last_time, symbol, interval),
    )
    conn.commit()


def _rebuild_daily_volatility(conn, symbol):
    """Sum 96 squared log returns for complete UTC days using the prior close."""
    conn.execute('DELETE FROM crypto_daily_volatility WHERE symbol = ?', (symbol,))
    rows = conn.execute(
        '''
        SELECT time, open, high, low, close, volume
        FROM crypto_intraday_history
        WHERE symbol = ?
        ORDER BY time
        ''',
        (symbol,),
    )

    output = []
    previous_close = None
    previous_timestamp = None
    current_date = None
    bars = clean_bars = returns = 0
    realized_variance = 0.0

    def finish_day():
        if current_date is not None and bars == clean_bars == returns == 96:
            output.append((symbol, current_date, realized_variance, bars))

    for timestamp, *values in rows:
        date = timestamp[:10]
        if date != current_date:
            finish_day()
            current_date = date
            bars = clean_bars = returns = 0
            realized_variance = 0.0
            expected_previous_date = (
                datetime.fromisoformat(date).date() - timedelta(days=1)
            ).isoformat()
            if (
                previous_timestamp is None
                or previous_timestamp[:10] != expected_previous_date
                or previous_timestamp[11:16] != '23:45'
                or timestamp[11:16] != '00:00'
            ):
                previous_close = None

        bars += 1
        clean = all(value is not None and math.isfinite(value) for value in values)
        clean_bars += clean
        close = values[3]
        if clean and close > 0 and previous_close is not None:
            realized_variance += math.log(close / previous_close) ** 2
            returns += 1
        previous_close = close if clean and close > 0 else None
        previous_timestamp = timestamp

    finish_day()
    conn.executemany(
        '''
        INSERT INTO crypto_daily_volatility
            (symbol, date, realized_variance, bars)
        VALUES (?, ?, ?, ?)
        ''',
        output,
    )


def _update_coverage(conn, symbol, from_time, to_time):
    valid = 'd.open IS NOT NULL AND d.high IS NOT NULL AND d.low IS NOT NULL AND d.close IS NOT NULL AND d.volume IS NOT NULL'
    from_date, to_date = from_time[:10], to_time[:10]
    daily_before, daily_after, first_date, last_date = conn.execute(
        '''
        SELECT COUNT(*),
               COALESCE(SUM(open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL AND volume IS NOT NULL), 0),
               MIN(substr(time, 1, 10)), MAX(substr(time, 1, 10))
        FROM crypto_daily_history
        WHERE symbol = ? AND substr(time, 1, 10) >= ? AND substr(time, 1, 10) < ?
        ''',
        (symbol, from_date, to_date),
    ).fetchone()
    intraday_days_before = conn.execute(
        '''
        SELECT COUNT(DISTINCT substr(time, 1, 10))
        FROM crypto_intraday_history
        WHERE symbol = ? AND substr(time, 1, 10) >= ? AND substr(time, 1, 10) < ?
        ''',
        (symbol, from_date, to_date),
    ).fetchone()[0]
    intraday_days_after = conn.execute(
        '''
        SELECT COUNT(*) FROM crypto_daily_volatility
        WHERE symbol = ? AND date >= ? AND date < ?
        ''',
        (symbol, from_date, to_date),
    ).fetchone()[0]
    aligned = conn.execute(
        f'''
        SELECT COUNT(*)
        FROM crypto_daily_history d
        JOIN crypto_daily_volatility v
          ON v.symbol = d.symbol AND v.date = substr(d.time, 1, 10)
        WHERE d.symbol = ? AND substr(d.time, 1, 10) >= ?
          AND substr(d.time, 1, 10) < ? AND {valid}
        ''',
        (symbol, from_date, to_date),
    ).fetchone()[0]
    expected_days = (
        datetime.fromisoformat(to_time.replace('Z', '+00:00')).date()
        - datetime.fromisoformat(from_time.replace('Z', '+00:00')).date()
    ).days
    conn.execute(
        'INSERT OR REPLACE INTO crypto_history_coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            symbol, daily_before, daily_after, intraday_days_before,
            intraday_days_after, aligned, int(aligned == expected_days), first_date, last_date,
        ),
    )


def scan_crypto_history(
    db_filename=HISTORY_DB,
    api_key=None,
    from_time=CRYPTO_FROM,
    to_time=CRYPTO_AS_OF,
    page_size=1_000,
    workers=3,
    log_filename=CRYPTO_LOG,
):
    """Fetch all altFINS symbols, daily bars, 15-minute bars, and daily RV."""
    with _session(api_key) as session:
        symbols = fetch_altfins_symbols(session=session)
    Path(db_filename).parent.mkdir(parents=True, exist_ok=True)
    logger = _scan_logger(log_filename)
    logger.info(
        'Starting/resuming %s symbols from %s through %s with %s workers',
        len(symbols), from_time, to_time, workers,
    )
    with closing(sqlite3.connect(db_filename)) as conn:
        _create_crypto_tables(conn)
        conn.execute('UPDATE crypto_symbols SET active = 0')
        conn.executemany(
            '''
            INSERT INTO crypto_symbols (symbol, friendly_name, active)
            VALUES (?, ?, 1)
            ON CONFLICT(symbol) DO UPDATE SET friendly_name = excluded.friendly_name, active = 1
            ''',
            symbols,
        )
        conn.commit()

    def scan_one(position, symbol):
        with _session(api_key) as worker_session, closing(
            sqlite3.connect(db_filename, timeout=60)
        ) as conn:
            conn.execute('PRAGMA busy_timeout = 60000')
            try:
                for interval in CRYPTO_INTERVALS:
                    _download_interval(
                        conn, worker_session, symbol, interval, from_time, to_time,
                        page_size,
                    )
                _rebuild_daily_volatility(conn, symbol)
                _update_coverage(conn, symbol, from_time, to_time)
                conn.commit()
                return position, symbol, 'success'
            except Exception as exc:
                status = _download_failure_status(str(exc))
                logger.exception('%s failed', symbol)
                conn.execute(
                    '''
                    UPDATE crypto_downloads
                    SET status = ?, last_error = ?, updated_at = ?
                    WHERE symbol = ? AND status = 'running'
                    ''',
                    (status, str(exc), datetime.now(timezone.utc).isoformat(), symbol),
                )
                conn.commit()
                return position, symbol, status

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(scan_one, position, symbol)
            for position, (symbol, _) in enumerate(symbols, start=1)
        ]
        for future in as_completed(futures):
            position, symbol, status = future.result()
            message = f'[{position}/{len(symbols)}] {symbol}: {status}'
            print(message, flush=True)
            logger.info(message)
    logger.info('Scan finished')


def crypto_history_summary(db_filename=HISTORY_DB, target_points=9_587_674):
    """Return and print crypto coverage before and after NA/completeness checks."""
    with closing(sqlite3.connect(db_filename)) as conn:
        _create_crypto_tables(conn)
        requested = conn.execute(
            'SELECT COUNT(*) FROM crypto_symbols WHERE active = 1'
        ).fetchone()[0]
        values = conn.execute('''
            SELECT
                COALESCE(SUM(daily_before_na), 0),
                COALESCE(SUM(daily_after_na), 0),
                COALESCE(SUM(intraday_days_before_na), 0),
                COALESCE(SUM(intraday_days_after_na), 0),
                COALESCE(SUM(aligned_days_after_na), 0),
                COALESCE(SUM(full_period), 0)
            FROM crypto_history_coverage c
            JOIN crypto_symbols s USING (symbol)
            WHERE s.active = 1
        ''').fetchone()
        statuses = dict(conn.execute('''
            SELECT status, COUNT(*)
            FROM crypto_downloads
            GROUP BY status
        '''))

    result = dict(zip((
        'daily_before_na', 'daily_after_na', 'intraday_days_before_na',
        'intraday_days_after_na', 'aligned_days_after_na', 'full_period_symbols',
    ), values))
    result.update({
        'requested_symbols': requested,
        'target_points': target_points,
        'enough_aligned_points': result['aligned_days_after_na'] >= target_points,
        'downloads': statuses,
    })
    print(result)
    return result


def write_crypto_history_summary(
    db_filename=HISTORY_DB,
    output_filename='ref/crypto_history_coverage.md',
    target_points=9_587_674,
):
    result = crypto_history_summary(db_filename, target_points)
    enough = 'yes' if result['enough_aligned_points'] else 'no'
    text = f'''# Crypto historical coverage summary

Coverage is measured from {CRYPTO_FROM[:10]} through {CRYPTO_AS_OF[:10]} (exclusive).
The comparison target is {target_points:,} stock observations.

| Measure | Points / symbols |
|---|---:|
| altFINS symbols requested | {result['requested_symbols']:,} |
| Daily rows before NA removal | {result['daily_before_na']:,} |
| Daily rows after NA removal | {result['daily_after_na']:,} |
| Intraday days before completeness checks | {result['intraday_days_before_na']:,} |
| Complete 15-minute realized-variance days | {result['intraday_days_after_na']:,} |
| Daily/proxy aligned days after checks | {result['aligned_days_after_na']:,} |
| Full-period symbols | {result['full_period_symbols']:,} |

Enough aligned observations to meet the target: **{enough}**.

Download statuses: `{result['downloads']}`
'''
    output = Path(output_filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    return result


if __name__ == '__main__':
    scan_crypto_history()
    write_crypto_history_summary()
