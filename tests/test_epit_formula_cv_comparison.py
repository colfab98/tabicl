from argparse import Namespace

import pandas as pd
import pytest

from scripts.epit_pipeline.compare_formula_cv import prediction_metrics, verify_prediction


def test_export_checks_query_order_and_reproduces_metrics():
    frame=pd.DataFrame({'task_row_index':[3,7,11], 'observed_mV':[100.,200.,400.], 'prediction_mV':[110.,230.,370.]})
    member={'query_task_row_indices':[3,7,11]}
    reference=prediction_metrics(frame)
    tolerance=Namespace(metric_atol=1e-5,metric_rtol=1e-5)
    verify_prediction(frame,member,reference,tolerance)
    with pytest.raises(ValueError,match='query order'):
        verify_prediction(frame.iloc[::-1],member,reference,tolerance)
    changed=frame.copy()
    changed.loc[0,'prediction_mV']+=100
    with pytest.raises(ValueError,match='reproduce saved metrics'):
        verify_prediction(changed,member,reference,tolerance)


def test_nonfinite_exports_are_rejected():
    frame=pd.DataFrame({'task_row_index':[1,2], 'observed_mV':[1.,2.], 'prediction_mV':[float('nan'),2.]})
    with pytest.raises(ValueError,match='Nonfinite'):
        verify_prediction(frame,{'query_task_row_indices':[1,2]}, {},Namespace(metric_atol=1e-5,metric_rtol=1e-5))
