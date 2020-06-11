import nose.tools

import skeleton.generate_lookup_array as generate_lookup


def _helper_template_working(arr, template, equate_to=0):
    template_test_class = generate_lookup.Templates(*tuple(arr))
    template_function = getattr(template_test_class, template)
    nose.tools.assert_equals(template_function(), equate_to)


def test_generate_lookup_array():
    assert not generate_lookup.generate_lookup_array(2).sum(), "first two config numbers should be delete"


def test_templates():
    templates = ['first_template', 'second_template', 'third_template', 'fourth_template', 'fifth_template',
                 'sixth_template', 'seventh_template', 'eighth_template', 'ninth_template', 'tenth_template',
                 'eleventh_template', 'twelveth_template', 'thirteenth_template', 'fourteenth_template']
    zeroes_test_case = [0] * 26
    ones_test_case = [1] * 26
    first_template_test_case = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    second_template_test_case = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    third_template_test_case = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    fourth_template_test_case = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    fifth_template_test_case = [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    sixth_template_test_case = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    seventh_template_test_case = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    eighth_template_test_case = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    ninth_template_test_case = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    tenth_template_test_case = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    eleventh_template_test_case = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    twelveth_template_test_case = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    thirteenth_template_test_case = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    fourteenth_template_test_case = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    template_cases = [first_template_test_case, second_template_test_case, third_template_test_case,
                      fourth_template_test_case, fifth_template_test_case, sixth_template_test_case,
                      seventh_template_test_case, eighth_template_test_case, ninth_template_test_case,
                      tenth_template_test_case, eleventh_template_test_case, twelveth_template_test_case,
                      thirteenth_template_test_case, fourteenth_template_test_case]

    for ith_template_case, template in enumerate(templates):
        _helper_template_working(ones_test_case, template)
        _helper_template_working(zeroes_test_case, template)
        _helper_template_working(template_cases[ith_template_case], template, 1)
