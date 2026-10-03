"""Cohort guard for raw-history RFSV parameters."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from argparse import Namespace
from unittest.mock import patch
import json
import os

import numpy as np
import pandas as pd

from models.variance_fit import rfsv_fit
from models.support_scripts.raw_statistical_fit import fit_rfsv_full


def test_rfsv_raw_cohort_guard():
    source = Path('imgs/roughness_analysis/global/train')
    with TemporaryDirectory() as directory:
        target = Path(directory)
        for name in ('roughness_summary.csv', 'roughness_moments.csv', 'roughness_zeta.csv'):
            copyfile(source / name, target / name)
        fit = rfsv_fit('garman-klass', target, expected_cohort='raw_history_2714')
        assert 0 < fit['H'] < .5 and fit['nu_squared'] > 0
        assert rfsv_fit('parkinson', target, expected_cohort='derived_25y')['nu_squared'] > 0
        for cohort in ('derived_25y', 'raw_history_1'):
            try:
                rfsv_fit('garman-klass', target, expected_cohort=cohort)
            except ValueError:
                pass
            else:
                raise AssertionError('mismatched RFSV cohort accepted')
        moments_path = target / 'roughness_moments.csv'
        moments = pd.read_csv(moments_path)
        moments.loc[moments.Table.eq('equity_garman_klass_variance'), 'Cohort'] = 'derived_25y'
        moments.to_csv(moments_path, index=False)
        try:
            rfsv_fit('garman-klass', target, expected_cohort='raw_history_2714')
        except ValueError:
            pass
        else:
            raise AssertionError('mismatched RFSV moments cohort accepted')
        copyfile(source / 'roughness_moments.csv', moments_path)
        summary_path = target / 'roughness_summary.csv'
        summary = pd.read_csv(summary_path)
        summary.loc[summary.Table.eq('equity_garman_klass_variance'), 'NuSquared'] *= 2
        summary.to_csv(summary_path, index=False)
        try:
            rfsv_fit('garman-klass', target, expected_cohort='raw_history_2714')
        except ValueError:
            pass
        else:
            raise AssertionError('mismatched RFSV nu-squared accepted')


def test_full_history_fit_calls_forward():
    with TemporaryDirectory() as directory:
        old = Path.cwd()
        os.chdir(directory)
        try:
            database = Path('source.db')
            database.write_bytes(b'fixture')
            rows = [{'Ticker': ticker, 'Date': pd.Timestamp('2015-01-01') + pd.Timedelta(days=i),
                     'Valid': True, 'RawVariance': float(i + 1) if ticker == 'A' else 0.,
                     'Variance': float(i + 1) if ticker == 'A' else 1.}
                    for ticker in ('A', 'B') for i in range(37)]
            frame = pd.DataFrame(rows)
            frame['Ticker'] = frame.Ticker.astype('category')
            parameters = {'H': .1, 'nu_squared': .2, 'lag_type': 'observation',
                          'max_lag': 400, 'forecast_window': 'full_positive_history',
                          'source': 'fixture'}
            args = Namespace(estimator='garman-klass', scope='global', ticker=None,
                             limit_rows=None, limit_tickers=None, window_size=20,
                             database=str(database))
            with patch('models.support_scripts.raw_statistical_fit.load_raw_neural', return_value=(frame, {}, 1.)), \
                 patch('models.variance_fit.rfsv_fit', return_value=parameters) as read_parameters:
                path = fit_rfsv_full(args)
            read_parameters.assert_called_once_with('garman-klass', expected_cohort='raw_history_2',
                                                    forecast_window='full_positive_history')
            saved = json.loads(path.read_text(encoding='utf-8'))
            assert saved['training_history_rows'] == 37
            assert saved['train_tickers_with_history'] == 1
            assert saved['train_tickers_without_history'] == 1
            assert saved['parameters'] == parameters
            assert np.isfinite(saved['floor'])
            assert not Path('inference/checkpoints/rfsv/garman-klass/global/raw_history_w20/fit.json').exists()
        finally:
            os.chdir(old)


if __name__ == '__main__':
    test_rfsv_raw_cohort_guard()
