import numpy as np
from kal2_control.state_machine import TurningDirection


def calculate_turningpath(turningdirection, is_driving_cw: bool):

    if turningdirection == TurningDirection.Left and is_driving_cw == False:
        start_point = np.array([4.0, 2.8])
        end_point = np.array([2.7, 2.4])
    elif turningdirection == TurningDirection.Right and is_driving_cw == False:
        start_point = np.array([4.0, 2.8])
        end_point = np.array([4.05, 4.2])
    elif turningdirection == TurningDirection.Left and is_driving_cw == True:
        start_point = np.array([2.7, 4.0])
        end_point = np.array([4.0, 4.2])
    elif turningdirection == TurningDirection.Right and is_driving_cw == True:
        start_point = np.array([2.7, 4.0])
        end_point = np.array([2.75, 2.4])
    else:
        raise ValueError(f"Invalid combination: {turningdirection}, {is_driving_cw}")

    x0, y0 = start_point
    x1, y1 = end_point

    distance = np.linalg.norm(end_point - start_point)
    Radius = distance / np.sqrt(2)

    kappa_sp = 1 / Radius
    kappa_ep = 1 / Radius

    A = np.array([[x0**3, x0**2, x0, 1], [x1**3, x1**2, x1, 1], [6 * x0, 2, 0, 0], [6 * x1, 2, 0, 0]])
    print("A", A)
    B = np.array([y0, y1, kappa_sp, kappa_ep])
    print("B", B)

    coefficients = np.linalg.solve(A, B)
    a, b, c, d = coefficients

    def polynomial(x):
        return a * x**3 + b * x**2 + c * x + d

    x_values = np.linspace(x0, x1, 100)
    # print("x", x_values)
    y_values = np.polyval(coefficients, x_values)  # polynomial(x_values)
    # print("y", y_values)
    path = np.array([x_values, y_values])
    # print("path_shape", path.shape)

    return path
