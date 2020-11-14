import cv2
import numpy as np


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def calculate_distance_point_line(centroid, inicio_linea, final_linea):
    d = np.abs(np.cross(final_linea-inicio_linea,centroid-inicio_linea)/np.linalg.norm(final_linea-inicio_linea))
    return d