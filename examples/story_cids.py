from pycid.core.cid import CID
from pycid.core.cpd import DecisionDomain, FunctionCPD, UniformRandomCPD


def get_introduced_bias() -> CID:

    cid = CID(
        [
            ("A", "X"),  # defining the graph's nodes and edges
            ("Z", "X"),
            ("Z", "Y"),
            ("X", "D"),
            ("X", "Y"),
            ("D", "U"),
            ("Y", "U"),
        ],
        decisions=["D"],
        utilities=["U"],
    )

    cpd_a = UniformRandomCPD("A", [0, 1])
    cpd_z = UniformRandomCPD("Z", [0, 1])
    cpd_x = FunctionCPD("X", lambda a, z: a * z)
    cpd_d = DecisionDomain("D", [0, 1])
    cpd_y = FunctionCPD("Y", lambda x, z: x + z)
    cpd_u = FunctionCPD("U", lambda d, y: -((d - y) ** 2))

    cid.add_cpds(cpd_a, cpd_d, cpd_z, cpd_x, cpd_y, cpd_u)
    return cid


# TODO add parameterization
def get_fitness_tracker() -> CID:
    cid = CID(
        [
            ("TD", "TF"),
            ("TF", "SC"),
            ("TF", "C"),
            ("EF", "EWD"),
            ("EWD", "C"),
            ("C", "F"),
            ("P", "D"),
            ("P", "SC"),
            ("P", "F"),
            ("SC", "C"),
            ("SC", "EWD"),
        ],
        decisions=["C"],
        utilities=["F"],
    )

    return cid


def get_car_accident_predictor() -> CID:
    cid = CID(
        [
            ("B", "N"),
            ("N", "AP"),
            ("N", "P"),
            ("P", "Race"),
            ("Age", "Adt"),
            ("Adt", "Race"),
            ("Race", "Accu"),
            ("M", "AP"),
            ("AP", "Accu"),
        ],
        decisions=["AP"],
        utilities=["Accu"],
    )

    return cid


def get_content_recommender() -> CID:
    cid = CID(
        [("O", "I"), ("O", "M"), ("M", "P"), ("P", "I"), ("I", "C"), ("P", "C")],
        decisions=["P"],
        utilities=["C"],
    )

    return cid


def get_content_recommender2() -> CID:
    cid = CID([("O", "M"), ("M", "P"), ("P", "I"), ("I", "C"), ("P", "C")], decisions=["P"], utilities=["C"])

    return cid


def get_modified_content_recommender() -> CID:
    cid = CID(
        [("O", "I"), ("O", "M"), ("M", "P"), ("P", "I"), ("P", "C"), ("M", "C")],
        decisions=["P"],
        utilities=["C"],
    )

    return cid


def get_grade_predictor() -> CID:
    cid = CID(
        [("R", "HS"), ("HS", "E"), ("HS", "P"), ("E", "Gr"), ("Gr", "Ac"), ("Ge", "P"), ("P", "Ac")],
        decisions=["P"],
        utilities=["Ac"],
    )

    return cid
