from opensoundscape import taxa


def test_get_species_list():
    assert len(taxa.get_species_list()) > 0


def test_sci_to_bn_common():
    sp = taxa.get_species_list()[0]
    taxa.sci_to_bn_common(sp)


def test_sci_to_xc_common():
    sp = taxa.get_species_list()[0]
    taxa.sci_to_xc_common(sp)


def test_xc_common_to_sci():
    common = taxa.sci_to_xc_common(taxa.get_species_list()[0])
    taxa.xc_common_to_sci(common)


def test_bn_common_to_sci():
    common = taxa.sci_to_bn_common(taxa.get_species_list()[0])
    taxa.bn_common_to_sci(common)


def test_common_to_sci():
    common = taxa.sci_to_bn_common(taxa.get_species_list()[0])
    taxa.common_to_sci(common)
