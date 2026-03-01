from ramp.state_analysis.util import split_name


def test_split_name():
    name = "D5/coolant/Plate_4/Meat/Piece:(0.0000e+00, -2.0700e+00, -1.6000e+01)"
    assert split_name(name)[2:] == (0, -2.07, -16.0)
