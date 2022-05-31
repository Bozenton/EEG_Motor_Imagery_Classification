def transform_to_standard(raw_ch_names: list, standard_ch_names_list: list = None):
    """
    Transform the names of channel to standard form
    Reference: https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)

    :param raw_ch_names:
    :param standard_ch_names_list:
    :return: a mapping dictionary
    """
    prefix = ['AF', 'FC', 'FT', 'CP', 'TP', 'PO',
              'Fp', 'F', 'T', 'P', 'O', 'C', 'I']

    mapping = {}
    for i, name in enumerate(raw_ch_names):
        if name[0:2].upper() in prefix:
            standard_name = name[0:2].upper() + name[2:].replace('.', '')
        elif name[0:2] in prefix:
            standard_name = name[0:2] + name[2:].replace('.', '')
            pass
        elif name[0:1] in prefix:
            standard_name = name[0:1] + name[1:].replace('.', '')
            pass
        else:
            raise ValueError('The name of channel is not standard according to International 10–20 system')
        mapping[name] = standard_name

    # check if there is repeated element:
    mapping_value_set = set(mapping.values())
    if len(mapping_value_set) != len(mapping.values()):
        raise ValueError('There are repeated element after transform.'
                         'The name of channel is not standard according to International 10–20 system')

    if standard_ch_names_list is not None:
        for _, name in enumerate(mapping.values()):
            if name not in standard_ch_names_list:
                raise ValueError('Transformed channel name {} is not in standard name list'.format(name))

    return mapping