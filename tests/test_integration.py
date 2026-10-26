"""Synthetic integration only: exercises 10-subject control flow, not EEG efficacy."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eeg'))
import eeg_train

class Integration(unittest.TestCase):
    def test_ten_subject_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'dummy';source.write_text('synthetic')
            rng=np.random.default_rng(7)
            def record(*args):
                y=np.tile([0,0,1,1,2,3,4],8)
                x=np.eye(5)[y]+rng.normal(0,.05,(len(y),5))
                return dict(y=y,baseline=x.copy(),enhanced=x.copy(),times=np.arange(len(y))*30,
                            valid=np.ones(len(y),bool),names=np.array(list('abcde')))
            with patch.object(sys,'argv',['eeg_train.py','--out',str(base/'out'),'--trees','5','--learning-curve']),patch.object(eeg_train,'get_nights',return_value=[(s,1,source,source) for s in range(10)]),patch.object(eeg_train,'load_record',side_effect=record):
                eeg_train.main()
            out=base/'out';self.assertEqual(json.loads((out/'completion.json').read_text())['status'],'completed')
            self.assertEqual(len(json.loads((out/'learning_curve.json').read_text())),4)
            self.assertFalse(json.loads((out/'metrics.json').read_text())['smoke_only'])
            self.assertTrue((out/'model.joblib').exists())
