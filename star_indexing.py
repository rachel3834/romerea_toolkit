def find_common_constant_stars(star_info):
    """
    Function to identify a set of stars that are not known to be variable,
    and that have data in all three filters

    Parameters:
        star_info  dict     Star array indices and boolean indicating whether or not a star is variable

    Returns:
        star_index  array   Indices of the subset of stars matching the selection criteria
    """

    common_consts = set(star_info["rp"][0][~star_info["rp"][1]]) \
                    & set(star_info["gp"][0][~star_info["gp"][1]]) \
                    & set(star_info["ip"][0][~star_info["ip"][1]])

    return common_consts
