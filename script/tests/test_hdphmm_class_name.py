"""Fits saved on main pickle the model class as utils.hdphmm.WeakLimitHMM; they must load here."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hdphmm import StickyHDPHMM


def test_main_class_name_unpickles_to_this_class():
    # protocol-0 GLOBAL opcode: the reference a pickled main-branch fit carries
    assert pickle.loads(b"cutils.hdphmm\nWeakLimitHMM\n.") is StickyHDPHMM


def test_instance_pickled_under_the_main_name_loads():
    model = StickyHDPHMM(n_components=2)
    data = pickle.dumps(model, protocol=4).replace(b"StickyHDPHMM", b"WeakLimitHMM")
    assert type(pickle.loads(data)) is StickyHDPHMM
