import re
from dataclasses import dataclass

# TURBOMOLE COSMO solvent-probe radius written as rsolv in $cosmo.
COSMO_RSOLV = 1.30


@dataclass(frozen=True)
class SolventDefinition:
    canonical_name: str
    epsilon: float
    aliases: tuple[str, ...] = ()


# Static dielectric constants. Chloroform is 4.800 so implicit solvent writes
# the $cosmo epsilon/rsolv block expected for that solvent.
SOLVENT_LIBRARY: dict[str, SolventDefinition] = {
    '1,1,1-trichloroethane': SolventDefinition('1,1,1-trichloroethane', 7.0826, ('methyl chloroform',)),
    '1,1,2-trichloroethane': SolventDefinition('1,1,2-trichloroethane', 9.4209),
    '1,1,1,2-tetrachloroethane': SolventDefinition('1,1,1,2-tetrachloroethane', 8.1895),
    '1,1,2,2-tetrachloroethane': SolventDefinition('1,1,2,2-tetrachloroethane', 8.36),
    '1,2-dibromoethane': SolventDefinition('1,2-dibromoethane', 4.9313, ('ethylene dibromide',)),
    '1,2-dichloroethane': SolventDefinition('1,2-dichloroethane', 10.125, ('dce', 'ethylene dichloride')),
    '1,2-dimethoxyethane': SolventDefinition('1,2-dimethoxyethane', 7.172, ('dme', 'glyme')),
    '1,2-ethanediol': SolventDefinition('1,2-ethanediol', 40.245, ('ethylene glycol',)),
    '1,2,4-trimethylbenzene': SolventDefinition('1,2,4-trimethylbenzene', 2.3653, ('pseudocumene',)),
    '1,3-dioxolane': SolventDefinition('1,3-dioxolane', 6.681),
    '1,4-dioxane': SolventDefinition('1,4-dioxane', 2.2099, ('dioxane',)),
    '1-bromo-2-methylpropane': SolventDefinition('1-bromo-2-methylpropane', 7.7796, ('isobutyl bromide',)),
    '1-bromooctane': SolventDefinition('1-bromooctane', 5.0244, ('bromooctane',)),
    '1-bromopentane': SolventDefinition('1-bromopentane', 6.269),
    '1-bromopropane': SolventDefinition('1-bromopropane', 8.0496),
    '1-butanol': SolventDefinition('1-butanol', 17.332, ('butanol', 'n-butanol')),
    '1-butyne': SolventDefinition('1-butyne', 2.3462),
    '1-chlorohexane': SolventDefinition('1-chlorohexane', 5.9491, ('chlorohexane',)),
    '1-chlorooctane': SolventDefinition('1-chlorooctane', 4.8867),
    '1-chloropentane': SolventDefinition('1-chloropentane', 6.5022),
    '1-chloropropane': SolventDefinition('1-chloropropane', 8.3548),
    '1-decanol': SolventDefinition('1-decanol', 7.5305, ('decanol',)),
    '1-fluorobutane': SolventDefinition('1-fluorobutane', 7.7147),
    '1-fluorodecane': SolventDefinition('1-fluorodecane', 3.6893),
    '1-fluorohexane': SolventDefinition('1-fluorohexane', 5.77),
    '1-fluoroheptane': SolventDefinition('1-fluoroheptane', 5.12),
    '1-fluorononane': SolventDefinition('1-fluorononane', 4.1044),
    '1-fluorooctane': SolventDefinition('1-fluorooctane', 3.89),
    '1-fluoropentane': SolventDefinition('1-fluoropentane', 6.6094),
    '1-heptanol': SolventDefinition('1-heptanol', 11.321, ('heptanol', 'n-heptanol')),
    '1-hexanol': SolventDefinition('1-hexanol', 12.51, ('hexanol', 'n-hexanol')),
    '1-hexene': SolventDefinition('1-hexene', 2.0717),
    '1-hexyne': SolventDefinition('1-hexyne', 2.615),
    '1-iodobutane': SolventDefinition('1-iodobutane', 6.1731),
    '1-iodohexadecane': SolventDefinition('1-iodohexadecane', 3.5338, ('hexadecyliodide',)),
    '1-iodohexane': SolventDefinition('1-iodohexane', 5.63),
    '1-iodoheptane': SolventDefinition('1-iodoheptane', 4.77),
    '1-iodooctane': SolventDefinition('1-iodooctane', 4.21),
    '1-iodopentane': SolventDefinition('1-iodopentane', 5.6973),
    '1-iodopropane': SolventDefinition('1-iodopropane', 6.9626),
    '1-nitropropane': SolventDefinition('1-nitropropane', 23.73),
    '1-nonanol': SolventDefinition('1-nonanol', 8.5991, ('nonanol', 'n-nonanol')),
    '1-octanol': SolventDefinition('1-octanol', 9.8629, ('octanol', 'n-octanol')),
    '1-pentanol': SolventDefinition('1-pentanol', 15.13, ('pentanol', 'n-pentanol', 'amyl alcohol')),
    '1-pentene': SolventDefinition('1-pentene', 1.9905),
    '1-propanol': SolventDefinition('1-propanol', 20.524, ('n-propanol', 'propanol', 'propyl alcohol')),
    '2,2,2-trifluoroethanol': SolventDefinition('2,2,2-trifluoroethanol', 26.726, ('tfe',)),
    '2,2,2-trichloroethanol': SolventDefinition('2,2,2-trichloroethanol', 27.91),
    '2,2,4-trimethylpentane': SolventDefinition('2,2,4-trimethylpentane', 1.9358, ('isooctane',)),
    '2,2-dimethoxypropane': SolventDefinition('2,2-dimethoxypropane', 3.72),
    '2,2-dimethylbutane': SolventDefinition('2,2-dimethylbutane', 1.9358),
    '2,3-dimethylbutane': SolventDefinition('2,3-dimethylbutane', 1.9848),
    '2,4-dimethylpentane': SolventDefinition('2,4-dimethylpentane', 1.8939),
    '2,4-dimethylpyridine': SolventDefinition('2,4-dimethylpyridine', 9.4176, ('2,4-lutidine',)),
    '2,6-dimethylpyridine': SolventDefinition('2,6-dimethylpyridine', 7.1735, ('2,6-lutidine',)),
    '2-bromopropane': SolventDefinition('2-bromopropane', 9.361, ('isopropyl bromide',)),
    '2-butanol': SolventDefinition('2-butanol', 15.944, ('sec-butanol',)),
    '2-chlorobutane': SolventDefinition('2-chlorobutane', 8.393, ('sec-butyl chloride',)),
    '2-heptanone': SolventDefinition('2-heptanone', 11.658, ('methyl amyl ketone',)),
    '2-hexanone': SolventDefinition('2-hexanone', 14.136, ('methyl butyl ketone',)),
    '2-methoxyethanol': SolventDefinition('2-methoxyethanol', 17.2, ('methyl cellosolve',)),
    '2-methyl-1-propanol': SolventDefinition('2-methyl-1-propanol', 16.777, ('isobutanol', 'isobutyl alcohol')),
    '2-methyl-2-propanol': SolventDefinition('2-methyl-2-propanol', 10.288, ('tert-butanol', 't-butanol')),
    '2-methylpentane': SolventDefinition('2-methylpentane', 1.89, ('isohexane',)),
    '2-methylpyridine': SolventDefinition('2-methylpyridine', 9.9533, ('2-picoline',)),
    '2-nitropropane': SolventDefinition('2-nitropropane', 25.654),
    '2-octanone': SolventDefinition('2-octanone', 9.4678),
    '2-pentanone': SolventDefinition('2-pentanone', 15.2, ('methyl propyl ketone',)),
    '2-propanol': SolventDefinition('2-propanol', 19.264, ('isopropanol', 'isopropyl alcohol', 'ipa')),
    '2-propen-1-ol': SolventDefinition('2-propen-1-ol', 19.011, ('allyl alcohol',)),
    '3-methylpyridine': SolventDefinition('3-methylpyridine', 11.645, ('3-picoline',)),
    '3-pentanone': SolventDefinition('3-pentanone', 16.78, ('diethyl ketone',)),
    '4-heptanone': SolventDefinition('4-heptanone', 12.257),
    '4-methyl-2-pentanone': SolventDefinition('4-methyl-2-pentanone', 12.887, ('mibk', 'methyl isobutyl ketone')),
    '4-methylpyridine': SolventDefinition('4-methylpyridine', 11.957, ('4-picoline',)),
    '5-nonanone': SolventDefinition('5-nonanone', 10.6),
    'benzaldehyde': SolventDefinition('benzaldehyde', 18.22),
    'benzene': SolventDefinition('benzene', 2.2706),
    'benzenethiol': SolventDefinition('benzenethiol', 4.2728, ('thiophenol',)),
    'benzonitrile': SolventDefinition('benzonitrile', 25.592),
    'benzyl alcohol': SolventDefinition('benzyl alcohol', 12.457),
    'benzyl benzoate': SolventDefinition('benzyl benzoate', 5.7425),
    'bicyclohexyl': SolventDefinition('bicyclohexyl', 2.3771),
    'bis(2-methoxyethyl) ether': SolventDefinition('bis(2-methoxyethyl) ether', 7.2, ('diglyme',)),
    'bromobenzene': SolventDefinition('bromobenzene', 5.3954),
    'bromoethane': SolventDefinition('bromoethane', 9.01, ('ethyl bromide',)),
    'bromoform': SolventDefinition('bromoform', 4.2488),
    'bromopentafluorobenzene': SolventDefinition('bromopentafluorobenzene', 8.7748),
    'butanal': SolventDefinition('butanal', 13.45),
    'butanoic acid': SolventDefinition('butanoic acid', 2.9931, ('butyric acid',)),
    'butanone': SolventDefinition('butanone', 18.246, ('methyl ethyl ketone', 'mek', '2-butanone')),
    'butanonitrile': SolventDefinition('butanonitrile', 24.291, ('butyronitrile',)),
    'butylamine': SolventDefinition('butylamine', 4.6178, ('n-butylamine',)),
    'butyl ethanoate': SolventDefinition('butyl ethanoate', 4.9941, ('butyl acetate',)),
    'carbon disulfide': SolventDefinition('carbon disulfide', 2.6105, ('cs2',)),
    'chlorobenzene': SolventDefinition('chlorobenzene', 5.6968),
    'chloroform': SolventDefinition('chloroform', 4.8, ('trichloromethane', 'chcl3')),
    'cis-decalin': SolventDefinition('cis-decalin', 2.2139),
    'cyclohexane': SolventDefinition('cyclohexane', 2.0165),
    'cyclohexanone': SolventDefinition('cyclohexanone', 15.619),
    'cyclopentanone': SolventDefinition('cyclopentanone', 13.58),
    'cyclopentanol': SolventDefinition('cyclopentanol', 16.989),
    'cyclopentyl methyl ether': SolventDefinition('cyclopentyl methyl ether', 4.7563, ('cpme',)),
    'decalin': SolventDefinition('decalin', 2.196),
    'dibromomethane': SolventDefinition('dibromomethane', 7.2273, ('methylene dibromide',)),
    'dibutyl ether': SolventDefinition('dibutyl ether', 3.0473),
    'diethyl ether': SolventDefinition('diethyl ether', 4.24, ('ether', 'et2o')),
    'diethyl sulfide': SolventDefinition('diethyl sulfide', 5.723, ('ethyl sulfide',)),
    'diethylamine': SolventDefinition('diethylamine', 3.5766),
    'diiodomethane': SolventDefinition('diiodomethane', 5.32, ('methylene iodide',)),
    'diisopropyl ether': SolventDefinition('diisopropyl ether', 3.38, ('isopropyl ether',)),
    'dimethyldisulfide': SolventDefinition('dimethyldisulfide', 9.6, ('dmds',)),
    'dimethylethanoamide': SolventDefinition('dimethylethanoamide', 37.781, ('dimethylacetamide', 'dma', 'dmac', 'n,n-dimethylacetamide')),
    'dimethyl sulfoxide': SolventDefinition('dimethyl sulfoxide', 46.826, ('dmso', 'dimethylsulfoxide')),
    'diphenyl ether': SolventDefinition('diphenyl ether', 3.73),
    'dipropylamine': SolventDefinition('dipropylamine', 2.9112),
    'eicosane': SolventDefinition('eicosane', 1.9427),
    'ethanethiol': SolventDefinition('ethanethiol', 6.667, ('ethyl mercaptan',)),
    'ethanol': SolventDefinition('ethanol', 24.852, ('etoh', 'ethyl alcohol')),
    'ethylbenzene': SolventDefinition('ethylbenzene', 2.4339),
    'ethylene carbonate': SolventDefinition('ethylene carbonate', 89.78),
    'ethyl ethanoate': SolventDefinition('ethyl ethanoate', 5.9867, ('ethyl acetate',)),
    'ethyl methanoate': SolventDefinition('ethyl methanoate', 8.331, ('ethyl formate',)),
    'ethyl phenyl ether': SolventDefinition('ethyl phenyl ether', 4.1797, ('phenetole',)),
    'fluorobenzene': SolventDefinition('fluorobenzene', 5.42),
    'formamide': SolventDefinition('formamide', 108.94),
    'heptane': SolventDefinition('heptane', 1.9113, ('n-heptane',)),
    'hexadecane': SolventDefinition('hexadecane', 2.0402, ('n-hexadecane',)),
    'hexanoic acid': SolventDefinition('hexanoic acid', 2.6, ('caproic acid',)),
    'iodobenzene': SolventDefinition('iodobenzene', 4.547),
    'iodoethane': SolventDefinition('iodoethane', 7.6177, ('ethyl iodide',)),
    'iodomethane': SolventDefinition('iodomethane', 6.865, ('methyl iodide',)),
    'isopropylbenzene': SolventDefinition('isopropylbenzene', 2.3712, ('cumene',)),
    'm-cresol': SolventDefinition('m-cresol', 12.44),
    'mesitylene': SolventDefinition('mesitylene', 2.265),
    'methanol': SolventDefinition('methanol', 32.613, ('meoh', 'methyl alcohol')),
    'methyl benzoate': SolventDefinition('methyl benzoate', 6.7367),
    'methylbutanoate': SolventDefinition('methylbutanoate', 5.5607, ('methyl butyrate',)),
    'methylcyclohexane': SolventDefinition('methylcyclohexane', 2.024),
    'methyl propanoate': SolventDefinition('methyl propanoate', 6.0777, ('methyl propionate',)),
    'n-methyl-2-pyrrolidinone': SolventDefinition('n-methyl-2-pyrrolidinone', 32.23, ('nmp',)),
    'n-decane': SolventDefinition('n-decane', 1.9846, ('decane',)),
    'n-dodecane': SolventDefinition('n-dodecane', 2.006, ('dodecane',)),
    'n-heptadecane': SolventDefinition('n-heptadecane', 2.0712, ('heptadecane',)),
    'n-nonane': SolventDefinition('n-nonane', 1.9605, ('nonane',)),
    'n-octane': SolventDefinition('n-octane', 1.9406, ('octane',)),
    'n-pentadecane': SolventDefinition('n-pentadecane', 2.0333, ('pentadecane',)),
    'n-pentane': SolventDefinition('n-pentane', 1.8371, ('pentane',)),
    'n-undecane': SolventDefinition('n-undecane', 1.991, ('undecane',)),
    'nitrobenzene': SolventDefinition('nitrobenzene', 34.809),
    'nitroethane': SolventDefinition('nitroethane', 28.29),
    'nitromethane': SolventDefinition('nitromethane', 36.562),
    'o-dichlorobenzene': SolventDefinition('o-dichlorobenzene', 9.9949, ('1,2-dichlorobenzene',)),
    'o-xylene': SolventDefinition('o-xylene', 2.5454),
    'p-isopropyltoluene': SolventDefinition('p-isopropyltoluene', 2.2322, ('p-cymene',)),
    'p-xylene': SolventDefinition('p-xylene', 2.2705),
    'pentanal': SolventDefinition('pentanal', 10.0),
    'pentanoic acid': SolventDefinition('pentanoic acid', 2.6924, ('valeric acid',)),
    'pentylamine': SolventDefinition('pentylamine', 4.201, ('amylamine',)),
    'perfluorobenzene': SolventDefinition('perfluorobenzene', 2.029, ('hexafluorobenzene',)),
    'propanal': SolventDefinition('propanal', 18.5),
    'propanenitrile': SolventDefinition('propanenitrile', 29.324, ('propionitrile', 'ethyl cyanide')),
    'propanoic acid': SolventDefinition('propanoic acid', 3.44, ('propionic acid',)),
    'propylamine': SolventDefinition('propylamine', 4.9912, ('n-propylamine',)),
    'sec-butylbenzene': SolventDefinition('sec-butylbenzene', 2.3446),
    'sulfolane': SolventDefinition('sulfolane', 43.962, ('tetrahydrothiophene-s,s-dioxide',)),
    'tetrahydrofuran': SolventDefinition('tetrahydrofuran', 7.4257, ('thf',)),
    'tetralin': SolventDefinition('tetralin', 2.771),
    'tert-butylbenzene': SolventDefinition('tert-butylbenzene', 2.3447),
    'toluene': SolventDefinition('toluene', 2.3741),
    'trans-decalin': SolventDefinition('trans-decalin', 2.1781),
    'tributylphosphate': SolventDefinition('tributylphosphate', 8.1781, ('tbp',)),
    'triethylamine': SolventDefinition('triethylamine', 2.3832, ('tea',)),
    'water': SolventDefinition('water', 78.355, ('h2o',)),
    'xylene mixture': SolventDefinition('xylene mixture', 2.3879, ('xylene', 'mixed xylene')),
    'acetone': SolventDefinition('acetone', 20.7),
    'acetonitrile': SolventDefinition('acetonitrile', 36.6, ('mecn', 'acn')),
    'ammonia': SolventDefinition('ammonia', 22.4, ('nh3',)),
    'carbon tetrachloride': SolventDefinition('carbon tetrachloride', 2.24, ('ccl4',)),
    'dichloromethane': SolventDefinition('dichloromethane', 9.08, ('methylene chloride', 'ch2cl2', 'dcm')),
    'dimethylformamide': SolventDefinition('dimethylformamide', 38.3, ('dmf', 'n,n-dimethylformamide')),
    'hexane': SolventDefinition('hexane', 1.89, ('n-hexane',)),
    'pyridine': SolventDefinition('pyridine', 12.5),
}


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


_SOLVENT_LOOKUP: dict[str, SolventDefinition] = {}
for _entry in SOLVENT_LIBRARY.values():
    _SOLVENT_LOOKUP[_normalize_key(_entry.canonical_name)] = _entry
    for _alias in _entry.aliases:
        _SOLVENT_LOOKUP[_normalize_key(_alias)] = _entry


def supported_solvent_names() -> tuple[str, ...]:
    return tuple(sorted(SOLVENT_LIBRARY))


def lookup_solvent(solvent: str) -> SolventDefinition:
    text = "" if solvent is None else str(solvent).strip()
    if not text:
        raise ValueError("Implicit solvent mode requires a solvent name.")
    entry = _SOLVENT_LOOKUP.get(_normalize_key(text))
    if entry is None:
        raise ValueError(
            f"Unsupported implicit solvent {solvent!r}. "
            "Choose a known solvent such as chloroform, or set solvent_mode to "
            "explicit and provide solvent_epsilon."
        )
    return entry


def cosmo_control_group(qm_input) -> list[str] | None:
    """Return the $cosmo control group for this calculation, or None for gas phase."""
    mode = getattr(qm_input.solvent_mode, "value", qm_input.solvent_mode)
    if mode == "none":
        return None
    if mode == "implicit":
        epsilon = lookup_solvent(qm_input.solvent).epsilon
        refind = None
    elif mode == "explicit":
        epsilon = qm_input.solvent_epsilon
        if epsilon is None:
            raise ValueError("Explicit solvent mode requires solvent_epsilon.")
        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError("solvent_epsilon must be > 0.")
        refind = qm_input.solvent_refind
        if refind is not None:
            refind = float(refind)
            if refind <= 0:
                raise ValueError("solvent_refind must be > 0 when provided.")
    else:
        raise ValueError(f"Unsupported solvent mode: {mode!r}.")
    lines = [
        "$cosmo",
        f"epsilon={float(epsilon):9.3f}",
        f"rsolv={COSMO_RSOLV:5.2f}",
    ]
    if refind is not None:
        lines.append(f"refind={refind:9.4f}")
    return lines
