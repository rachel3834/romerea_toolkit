import unittest
#import indiv
import star_indexing
import numpy as np
class TestIndexing(unittest.TestCase):

    def test_find_common_constant_stars(self):

        # First we simulate the expected input data for this function
        # This has the added benefit of defining exactly what data we expect
        # to go into the function for future reference
        nstars = 10
        star_index = np.arange(0, nstars, 1)
        known_vars = np.empty(nstars, dtype=bool)
        known_vars.fill(False)
        var_star = 2
        known_vars[var_star] = True    # Make sure we flag a specific known variable star
        star_info = {
            'gp': (star_index, known_vars),     # If the input files have entries for all stars then
            'rp': (star_index, known_vars),     # the data for all filters should be the same
            'ip': (star_index, known_vars)      # Variable stars should be flagged regardless of filter
        }

        # Now we send this data through the function to be tested.
        # It helps to have code that is structured (or "refactored") into small distinct
        # functions as much as possible, because then you can test each piece separately.
        result = star_indexing.find_common_constant_stars(star_info)

        # At this point we have the results from the function and since we controlled
        # the input data, we also know what the results should be.  So now we test to verify
        # that's what we actually got.

        # Since we flagged one star (var_star) as variable, we know this function should have
        # excluded that star from the index.  So the length of the list should be shorter
        assert(len(result) == len(star_index)-1)

        # ...and it should not contain var_star
        assert(var_star not in list(result))
