import matplotlib.pyplot as plt
import numpy as np

def find_peaks_indices(x_array: np.array, y_array: np.array, eps: float=1e-12, atol: float=1e-3) -> np.array:
    def find_zero_crossings(array: np.array):

        # Find indices where the sign of consecutive elements changes
        sign_changes = np.where(np.diff(np.sign(array)))[0]

        # Adjust indices to account for the shifted array
        zero_crossings = sign_changes + 1

        return zero_crossings

    # Find zeros of first derivative
    first_derivative = np.gradient(y_array,
                                   # np.log(x_array)
                                   )
    first_deriv_zeros_mask = np.zeros_like(y_array, dtype=bool)

    # This detects very prominent peaks: where the variation is very fast and the derivative has an undoubted
    # change in sign.
    first_deriv_zeros_mask[find_zero_crossings(array=first_derivative)] = True

    # TODO: give a meaning to this tolerance, e.g. propagation of errors
    # This detects regions in the domain where the function might either have a peaks or a plateau if we admit a
    # small computational error.
    zero_indices = np.where(np.isclose(first_derivative, 0, atol=atol))[0]
    first_deriv_zeros_mask[zero_indices] = True

    # Calculate the second derivative
    second_derivative = np.gradient(first_derivative)

    # plt.semilogx(x_array[1:], first_derivative, label="1rst derivative")
    # plt.semilogx(x_array[1:], second_derivative, label='2nd derivative')
    # for i in np.where(first_deriv_zeros_mask)[0]:
    #     plt.axvline(x_array[i], linestyle='--', c='red')
    # # # plt.ylim(bottom=-1e-12, top=1e-12)
    # plt.xlim(left=1, right=1e3)
    # plt.ylim((-.001, .001))
    # plt.legend(loc=1)
    # plt.grid()
    # plt.show()
    # # plt.semilogx(x_array[1:], first_derivative, label="")
    # plt.show()

    # # Identify the index corresponding to the last peak for growing x-values
    # peak_indices = np.where(first_deriv_zeros_mask & (second_derivative < -eps))[0]
    # peak_indices = np.where(first_deriv_zeros_mask & (second_derivative < -eps) & (y_array > 5e-2))[0]
    peak_indices = np.where(first_deriv_zeros_mask & (second_derivative < -eps) & (y_array > .4))[0]

    # peak_indices = np.where(first_deriv_zeros_mask & (y_array > 1e-2))[0]

    if len(peak_indices) > 0:
        return peak_indices
    else:
        return None  # No peak found


# def find_first_peak_x(x_array, y_array) -> np.array:
#     def find_zero_crossings(array: np.array):
#
#         # Find indices where the sign of consecutive elements changes
#         sign_changes = np.where(np.diff(np.sign(array)))[0]
#
#         # Adjust indices to account for the shifted array
#         zero_crossings = sign_changes + 1
#
#         return zero_crossings
#
#     # Find zeros of first derivative
#     first_derivative = np.gradient(y_array)
#     first_deriv_zeros_mask = np.zeros_like(y_array, dtype=bool)
#     first_deriv_zeros_mask[find_zero_crossings(array=first_derivative)] = True
#
#     # Calculate the second derivative
#     second_derivative = np.gradient(first_derivative)
#
#     # Identify the index corresponding to the first peak for growing x-values
#     peak_index = np.where(first_deriv_zeros_mask & (second_derivative < 0))[0].min()
#     if peak_index > 0:
#         return x_array[1:][peak_index]
#     else:
#         return None  # No peak found
