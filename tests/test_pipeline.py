import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eeg'))
from data import acquire
from features import trailing_context
from transitions import detect,evaluate
from eeg_train import split_subjects

class PipelineTests(unittest.TestCase):
    def test_subject_split(self):
        s=split_subjects(list(range(10)),42)
        self.assertEqual([len(s[k]) for k in ['train','validation','test']],[6,2,2])
        self.assertEqual(len(set(sum(s.values(),[]))),10)
        self.assertEqual(s,split_subjects(list(range(10)),42))
    def test_future_invariance_and_gap(self):
        x=np.arange(12).reshape(6,2).astype(float);t=np.array([0,30,60,120,150,180]);valid=np.ones(6,dtype=bool)
        a=trailing_context(x,t,valid);x[4:]+=999
        b=trailing_context(x,t,valid)
        np.testing.assert_array_equal(a[:4],b[:4])
        np.testing.assert_array_equal(a[3,2:4],x[3]);np.testing.assert_array_equal(a[3,4:],0)
    def test_alert_end_not_backdated(self):
        y=np.array([0,0,1,1,2]);p=np.eye(5)[y];t=np.arange(5)*30;v=np.ones(5,dtype=bool)
        self.assertEqual(detect(p,t,v),[3])
        score=evaluate(y,p,t,v)
        self.assertEqual(score['median_delay_s'],60)
        self.assertEqual(score['matched'],1)
        v[2]=False;self.assertEqual(detect(p,t,v),[])
    def test_false_alert_and_no_reference(self):
        y=np.zeros(4,dtype=int);p=np.eye(5)[[0,0,1,1]];t=np.arange(4)*30
        score=evaluate(y,p,t,np.ones(4,dtype=bool))
        self.assertEqual(score['matched'],0);self.assertIsNone(score['recall'])
        self.assertEqual(score['unmatched_alerts_per_valid_hour'],30)
    def test_download_retry_and_cache(self):
        body=b'example data';row={'fname':'test.edf','sha':hashlib.sha1(body).hexdigest()}
        with tempfile.TemporaryDirectory() as d:
            with patch('data.urllib.request.urlopen',side_effect=[OSError('timeout'),io.BytesIO(body)]) as call, patch('data.time.sleep'):
                file=acquire(row,d,retries=2)
                self.assertEqual(call.call_count,2);self.assertEqual(file.read_bytes(),body)
            with patch('data.urllib.request.urlopen',side_effect=AssertionError('network accessed')):
                self.assertEqual(acquire(row,d,offline=True),file)
            file.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):acquire(row,d,offline=True)
    def test_keyboard_interrupt_not_swallowed(self):
        row={'fname':'test.edf','sha':'0'*40}
        with tempfile.TemporaryDirectory() as d, patch('data.urllib.request.urlopen',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):acquire(row,d)
    def test_failed_download_never_becomes_cache(self):
        row={'fname':'test.edf','sha':'0'*40}
        with tempfile.TemporaryDirectory() as d, patch('data.urllib.request.urlopen',return_value=io.BytesIO(b'incomplete')):
            with self.assertRaises(RuntimeError):acquire(row,d,retries=1)
            self.assertFalse((Path(d)/'physionet-sleep-data'/'test.edf').exists())

if __name__=='__main__':unittest.main()
