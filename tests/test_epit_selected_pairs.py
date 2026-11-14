import numpy as np
import pandas as pd
import pytest
from scripts.epit_pipeline.prepare_selected_pairs import compare_pairs


def test_selection_uses_absolute_error_and_discards_ties():
    generic=pd.DataFrame({'task_row_index':[0,1], 'observed_mV':[100.,200.], 'prediction_mV':[120.,190.], 'cv_fold':[1,2]})
    formula=pd.DataFrame({'task_row_index':[0,0,1,1], 'formula':['a','b','a','b'],
                          'observed_mV':[100.,100.,200.,200.], 'prediction_mV':[110.,80.,215.,205.]})
    result=compare_pairs(formula,generic)
    assert result.retained.tolist()==[True,False,False,True]
    assert result.retained.mean()==.5
    with pytest.raises(ValueError,match='Duplicate'):
        compare_pairs(formula,pd.concat([generic,generic]))
    bad=generic.copy();bad.loc[0,'observed_mV']=101
    with pytest.raises(AssertionError):
        compare_pairs(formula,bad)
    bad=formula.copy();bad.loc[0,'prediction_mV']=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        compare_pairs(bad,generic)
