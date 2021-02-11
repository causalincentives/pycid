from core.cid import CID
from core.cpd import UniformRandomCPD, FunctionCPD, DecisionDomain


def get_introduced_bias() -> CID:

    cid = CID([
        ('A', 'X'),  # defining the graph's nodes and edges
        ('Z', 'X'),
        ('Z', 'Y'),
        ('X', 'D'),
        ('X', 'Y'),
        ('D', 'U'),
        ('Y', 'U')
    ],
        decision_nodes=['D'],
        utility_nodes=['U'])

    cpd_a = UniformRandomCPD('A', [0, 1])
    cpd_z = UniformRandomCPD('Z', [0, 1])
    cpd_x = FunctionCPD('X', lambda a, z: a*z, evidence=['A', 'Z'])
    cpd_d = DecisionDomain('D', [0, 1])
    cpd_y = FunctionCPD('Y', lambda x, z: x + z, evidence=['X', 'Z'])
    cpd_u = FunctionCPD('U', lambda d, y: -(d - y) ** 2, evidence=['D', 'Y'])

    cid.add_cpds(cpd_a, cpd_d, cpd_z, cpd_x, cpd_y, cpd_u)
    return cid


# TODO add parameterization
def get_fitness_tracker() -> CID:
    cid = CID([
        ('TD', 'TF'),
        ('TF', 'SC'),
        ('TF', 'C'),
        ('EF', 'EWD'),
        ('EWD', 'C'),
        ('C', 'F'),
        ('P', 'D'),
        ('P', 'SC'),
        ('P', 'F'),
        ('SC', 'C'),
        ('SC', 'EWD'),
        ],
        decision_nodes=['C'],
        utility_nodes=['F'])

    return cid


def get_car_accident_predictor() -> CID:
    cid = CID([
        ('B', 'N'),
        ('N', 'AP'),
        ('N', 'P'),
        ('P', 'Race'),
        ('Age', 'Adt'),
        ('Adt', 'Race'),
        ('Race', 'Accu'),
        ('M', 'AP'),
        ('AP', 'Accu'),
        ],

        decision_nodes=['AP'],
        utility_nodes=['Accu']
        )

    return cid


def get_content_recommender() -> CID:
    cid = CID([
        ('O', 'I'),
        ('O', 'M'),
        ('M', 'P'),
        ('P', 'I'),
        ('I', 'C'),
        ('P', 'C'),
        ],
        decision_nodes=['P'],
        utility_nodes=['C']
        )

    return cid

def get_content_recommender2() -> CID:
    cid = CID([
        ('O', 'M'),
        ('M', 'P'),
        ('P', 'I'),
        ('I', 'C'),
        ('P', 'C'),
        ],
        decision_nodes=['P'],
        utility_nodes=['C']
        )

    return cid


def get_modified_content_recommender() -> CID:
    cid = CID([
        ('O', 'I'),
        ('O', 'M'),
        ('M', 'P'),
        ('P', 'I'),
        ('P', 'C'),
        ('M', 'C')
        ],
        decision_nodes=['P'],
        utility_nodes=['C']
    )

    return cid


def get_grade_predictor() -> CID:
    cid = CID([
        ('R', 'HS'),
        ('HS', 'E'),
        ('HS', 'P'),
        ('E', 'Gr'),
        ('Gr', 'Ac'),
        ('Ge', 'P'),
        ('P', 'Ac'),
        ],
        decision_nodes=['P'],
        utility_nodes=['Ac']
        )

    return cid
