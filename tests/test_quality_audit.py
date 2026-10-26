import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'eeg'))
from audit_quality import metrics, longest, read_header, discover, CHANNELS


def fixture(path):
    def field(x, n): return str(x).encode().ljust(n)
    head = b''.join([field('0',8),field('',80),field('',80),field('01.01.00',8),field('00.00.00',8),field(1280,8),field('',44),field(1,8),field(30,8),field(4,4)])
    values = [(CHANNELS,16),(['']*4,80),(['uV']*4,8),([-100]*4,8),([100]*4,8),([-2048]*4,8),([2047]*4,8),(['']*4,80),([3000,3000,3000,30],8),(['']*4,32)]
    path.write_bytes(head+b''.join(field(x,w) for xs,w in values for x in xs)+np.zeros(9030,dtype='<i2').tobytes())


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.c = dict(label=CHANNELS[0],unit='uV',pmin=-100,pmax=100,dmin=-100,dmax=100,hz=100)
    def test_units_and_scaling(self):
        d = np.array([-100,0,100])
        self.assertEqual(metrics(d,self.c)['ptp_uv'],200)
        self.c.update(unit='mV',pmin=-.1,pmax=.1)
        self.assertAlmostEqual(metrics(d,self.c)['ptp_uv'],200)
    def test_same_side_run(self):
        m = metrics(np.array([-100,100,-100,100,100,0]),self.c)
        self.assertEqual(m['rail_samples'],5)
        self.assertEqual(m['longest_same_rail_seconds'],.02)
    def test_flat_and_emg(self):
        self.assertTrue(metrics(np.zeros(30),self.c)['legacy_flat'])
        self.c['label']=CHANNELS[3]
        self.assertFalse(metrics(np.zeros(30),self.c)['legacy_flat'])
    def test_threshold_boundary(self):
        self.c.update(pmin=-500,pmax=500)
        self.assertTrue(metrics(np.array([-100,100]),self.c)['legacy_high_amplitude'])
    def test_outside_header(self):
        self.assertEqual(metrics(np.array([-101,101,0]),self.c)['out_of_header_samples'],2)
    def test_runs(self):
        self.assertEqual(longest([True,True,False,True]),2)
        self.assertEqual(longest([False]),0)
    def test_header_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'SC4001E0-PSG.edf'; fixture(p)
            h=read_header(p)
            self.assertEqual(h['channels'][3]['hz'],1)
            self.assertEqual(h['channels'][3]['offset'],9000)
            self.assertEqual(discover(tmp,[0]),[(0,p)])
            p.write_bytes(p.read_bytes()[:-2])
            with self.assertRaises(ValueError): read_header(p)
    def test_missing_duplicate_and_unknown_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'SC4001E0-PSG.edf'; fixture(p)
            with self.assertRaises(ValueError): discover(tmp,[1])
            other=Path(tmp)/'copy'; other.mkdir(); fixture(other/p.name)
            with self.assertRaises(ValueError): discover(tmp,[0])
            b=bytearray(p.read_bytes()); b[640:648]=b'unknown '
            p.write_bytes(b)
            with self.assertRaises(ValueError): read_header(p)

if __name__=='__main__': unittest.main()
