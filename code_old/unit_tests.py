import numpy as np

import fcm

def test_face_bisection():
    # definition of the domain
    center = [0.0, 0.0]
    radius = np.pi/4
    domain = lambda x, y : (x - center[0])**2 + (y - center[1])**2 >= radius**2

    a = np.array([0, 0])
    b = np.array([1, 0])

    c = fcm.face_bisection(domain, a, b,)

    print(f"Position of cut: ({radius}, 0.0)")
    print(f"Computed: ({c[0]}, {c[1]})")

    return


test_face_bisection()