"""
Gesture FX Showcase — Star-Dome Planetarium
=============================================
Raise a hand into frame: the room gently fades toward a dim, cool-toned
night sky as the major constellations fade in around you, as if you were
standing at the center of a giant sky globe — background stars twinkle,
bright stars glow softly. Pinch (thumb + index finger together) and drag
to rotate the dome and look around. Release the pinch to stop rotating.
Drop the hand out of frame and it all fades back to the normal camera feed.

Controls
--------
  Pinch + drag (one hand)   -> rotate the dome (yaw / pitch)
  Spread / bring together
    two hands               -> zoom in / out (bigger sphere feel <-> smaller)
  f                         -> toggle fullscreen
  q                         -> quit

Requirements
------------
  pip install opencv-python mediapipe numpy

This uses MediaPipe's current Tasks API (mediapipe>=0.10). The older
`mp.solutions.hands` module is deprecated and no longer ships in recent
mediapipe releases, so hand tracking below uses `HandLandmarker` instead.
On first run the script downloads the small hand-landmark model file
(hand_landmarker.task, ~8 MB) into the same folder and reuses it after
that — an internet connection is only needed the first time.

Notes on accuracy
------------------
Star positions are real J2000 equatorial coordinates (right ascension in
hours, declination in degrees) for each constellation's principal stars,
taken to ~3 decimal places — accurate enough to reproduce each
constellation's true relative shape and its position against its neighbors.
Star sizes are driven by real approximate apparent magnitudes, so brighter
stars (Sirius, Vega, Rigel...) are visibly bigger/brighter than faint ones,
instead of every star rendering the same size. A handful of stars that are
genuine orange/red giants in reality (Betelgeuse, Antares, Arcturus...) get
a warm tint; everything else renders cool blue-white. This is a
visualization, not a survey instrument, so proper motion, precession, and
minor stars are left out on purpose.
"""

import math
import os
import random
import time
import urllib.request
import urllib.error
import http.client
import cv2
import numpy as np
import mediapipe as mp

# --------------------------------------------------------------------------
# 1. Constellation data — (RA in hours, Dec in degrees), J2000
# --------------------------------------------------------------------------
CONSTELLATIONS = {
    "Ursa Major": {
        "stars": {
            "Dubhe":  (11.062, 61.751), "Merak":  (11.031, 56.382),
            "Phecda": (11.897, 53.695), "Megrez": (12.257, 57.033),
            "Alioth": (12.900, 55.960), "Mizar":  (13.399, 54.925),
            "Alkaid": (13.792, 49.313),
        },
        "lines": [("Dubhe", "Merak"), ("Merak", "Phecda"), ("Phecda", "Megrez"),
                  ("Megrez", "Dubhe"), ("Megrez", "Alioth"), ("Alioth", "Mizar"),
                  ("Mizar", "Alkaid")],
    },
    "Ursa Minor": {
        "stars": {
            "Polaris": (2.530, 89.264), "Yildun":  (17.537, 86.586),
            "EpsUMi":  (16.766, 82.037), "ZetaUMi": (15.734, 77.794),
            "Kochab":  (14.845, 74.156), "Pherkad": (15.345, 71.834),
            "EtaUMi":  (16.292, 75.755),
        },
        "lines": [("Polaris", "Yildun"), ("Yildun", "EpsUMi"), ("EpsUMi", "ZetaUMi"),
                  ("ZetaUMi", "Kochab"), ("Kochab", "Pherkad"), ("Pherkad", "EtaUMi"),
                  ("EtaUMi", "ZetaUMi")],
    },
    "Cassiopeia": {
        "stars": {
            "Caph": (0.153, 59.150), "Schedar": (0.675, 56.537),
            "Navi": (0.945, 60.717), "Ruchbah": (1.430, 60.235),
            "Segin": (1.906, 63.670),
        },
        "lines": [("Caph", "Schedar"), ("Schedar", "Navi"), ("Navi", "Ruchbah"),
                  ("Ruchbah", "Segin")],
    },
    "Orion": {
        "stars": {
            "Betelgeuse": (5.919, 7.407), "Bellatrix": (5.418, 6.350),
            "Mintaka": (5.533, -0.299), "Alnilam": (5.603, -1.202),
            "Alnitak": (5.679, -1.943), "Saiph": (5.796, -9.670),
            "Rigel": (5.242, -8.202),
        },
        "lines": [("Betelgeuse", "Bellatrix"), ("Bellatrix", "Mintaka"),
                  ("Mintaka", "Alnilam"), ("Alnilam", "Alnitak"),
                  ("Alnitak", "Saiph"), ("Saiph", "Rigel"), ("Rigel", "Mintaka"),
                  ("Betelgeuse", "Alnitak")],
    },
    "Cygnus": {
        "stars": {
            "Deneb": (20.690, 45.280), "Sadr": (20.370, 40.257),
            "Gienah": (20.770, 33.970), "DeltaCyg": (19.749, 45.131),
            "Albireo": (19.512, 27.960),
        },
        "lines": [("Deneb", "Sadr"), ("Sadr", "Gienah"), ("Sadr", "DeltaCyg"),
                  ("Sadr", "Albireo")],
    },
    "Lyra": {
        "stars": {
            "Vega": (18.615, 38.784), "Sheliak": (18.835, 33.363),
            "Sulafat": (18.982, 32.690), "DeltaLyr": (18.897, 36.898),
        },
        "lines": [("Vega", "DeltaLyr"), ("DeltaLyr", "Sheliak"),
                  ("Sheliak", "Sulafat"), ("Sulafat", "DeltaLyr")],
    },
    "Leo": {
        "stars": {
            "Regulus": (10.139, 11.967), "Algieba": (10.333, 19.842),
            "Zosma": (11.235, 20.524), "Denebola": (11.818, 14.572),
            "Chertan": (11.237, 15.430),
        },
        "lines": [("Regulus", "Algieba"), ("Algieba", "Zosma"),
                  ("Zosma", "Chertan"), ("Chertan", "Denebola"),
                  ("Chertan", "Regulus")],
    },
    "Scorpius": {
        "stars": {
            "Dschubba": (16.005, -22.622), "Graffias": (16.090, -19.805),
            "Antares": (16.490, -26.432), "Sargas": (17.622, -42.998),
            "Shaula": (17.560, -37.104), "Lesath": (17.513, -37.296),
        },
        "lines": [("Graffias", "Dschubba"), ("Dschubba", "Antares"),
                  ("Antares", "Sargas"), ("Sargas", "Shaula"), ("Shaula", "Lesath")],
    },
    "Taurus": {
        "stars": {
            "Aldebaran": (4.599, 16.509), "Elnath": (5.438, 28.608),
            "Zeta_Tau": (5.629, 21.143), "Hyadum": (4.478, 15.872),
        },
        "lines": [("Elnath", "Zeta_Tau"), ("Zeta_Tau", "Aldebaran"),
                  ("Aldebaran", "Hyadum")],
    },
    "Gemini": {
        "stars": {
            "Castor": (7.577, 31.888), "Pollux": (7.755, 28.026),
            "Alhena": (6.629, 16.399), "Wasat": (7.335, 21.982),
            "Mebsuta": (6.732, 25.131),
        },
        "lines": [("Castor", "Pollux"), ("Castor", "Mebsuta"),
                  ("Mebsuta", "Alhena"), ("Pollux", "Wasat"), ("Wasat", "Alhena")],
    },
    "Canis Major": {
        "stars": {
            "Sirius": (6.752, -16.716), "Mirzam": (6.378, -17.956),
            "Wezen": (7.140, -26.393), "Adhara": (6.977, -28.972),
        },
        "lines": [("Mirzam", "Sirius"), ("Sirius", "Adhara"), ("Adhara", "Wezen")],
    },
    "Bootes": {
        "stars": {
            "Arcturus": (14.261, 19.182), "Izar": (14.749, 27.074),
            "Seginus": (14.535, 38.308), "Nekkar": (15.032, 40.391),
            "Muphrid": (13.918, 18.398),
        },
        "lines": [("Muphrid", "Arcturus"), ("Arcturus", "Izar"),
                  ("Izar", "Seginus"), ("Seginus", "Nekkar")],
    },
    "Andromeda": {
        "stars": {
            "Alpheratz": (0.139, 29.090), "Mirach": (1.162, 35.621),
            "Almach": (2.065, 42.330),
        },
        "lines": [("Alpheratz", "Mirach"), ("Mirach", "Almach")],
    },
    "Auriga": {
        "stars": {
            "Capella": (5.278, 45.998), "Menkalinan": (5.992, 44.947),
            "Elnath2": (5.438, 28.608), "Hassaleh": (4.950, 33.166),
            "Almaaz": (5.038, 43.823),
        },
        "lines": [("Capella", "Menkalinan"), ("Menkalinan", "Elnath2"),
                  ("Elnath2", "Hassaleh"), ("Hassaleh", "Almaaz"), ("Almaaz", "Capella")],
    },
    "Pegasus": {
        "stars": {
            "Markab": (23.079, 15.205), "Scheat": (23.063, 28.083),
            "Algenib": (0.221, 15.183), "Alpheratz2": (0.139, 29.090),
        },
        "lines": [("Markab", "Scheat"), ("Scheat", "Alpheratz2"),
                  ("Alpheratz2", "Algenib"), ("Algenib", "Markab")],
    },
    "Perseus": {
        "stars": {
            "Mirfak": (3.405, 49.861), "Algol": (3.136, 40.956),
            "Atik": (3.902, 31.884), "Miram": (2.844, 55.895),
        },
        "lines": [("Miram", "Mirfak"), ("Mirfak", "Algol"), ("Algol", "Atik")],
    },
    "Cepheus": {
        "stars": {
            "Alderamin": (21.310, 62.585), "Alfirk": (21.478, 70.561),
            "Errai": (23.657, 77.632), "ZetaCep": (22.166, 58.202),
        },
        "lines": [("ZetaCep", "Alderamin"), ("Alderamin", "Alfirk"), ("Alfirk", "Errai")],
    },
    "Draco": {
        "stars": {
            "Thuban": (14.073, 64.376), "Rastaban": (17.507, 52.301),
            "Eltanin": (17.943, 51.489), "Altais": (19.209, 67.661),
        },
        "lines": [("Thuban", "Rastaban"), ("Rastaban", "Eltanin"), ("Eltanin", "Altais")],
    },
    "Hercules": {
        "stars": {
            "Kornephoros": (16.503, 21.489), "ZetaHer": (16.688, 31.603),
            "PiHer": (17.250, 36.809), "EtaHer": (16.715, 38.922),
        },
        "lines": [("Kornephoros", "ZetaHer"), ("ZetaHer", "EtaHer"),
                  ("EtaHer", "PiHer"), ("PiHer", "ZetaHer")],
    },
    "Corona Borealis": {
        "stars": {
            "Alphecca": (15.578, 26.715), "Nusakan": (15.462, 29.106),
            "ThetaCrB": (15.550, 31.360), "GammaCrB": (15.567, 26.295),
        },
        "lines": [("Nusakan", "Alphecca"), ("Alphecca", "GammaCrB"), ("GammaCrB", "ThetaCrB")],
    },
    "Aquila": {
        "stars": {
            "Altair": (19.846, 8.868), "Tarazed": (19.771, 10.613),
            "Alshain": (19.921, 6.407),
        },
        "lines": [("Tarazed", "Altair"), ("Altair", "Alshain")],
    },
    "Sagittarius": {
        "stars": {
            "KausAustralis": (18.403, -34.384), "KausMedia": (18.350, -29.828),
            "KausBorealis": (18.310, -25.421), "Alnasl": (18.083, -30.424),
            "Nunki": (18.921, -26.297), "Ascella": (18.921, -29.880),
        },
        "lines": [("Alnasl", "KausMedia"), ("KausMedia", "KausAustralis"),
                  ("KausAustralis", "Ascella"), ("Ascella", "Nunki"),
                  ("KausMedia", "KausBorealis")],
    },
    "Capricornus": {
        "stars": {
            "Algedi": (20.301, -12.545), "Dabih": (20.350, -14.781),
            "DenebAlgedi": (21.784, -16.127),
        },
        "lines": [("Algedi", "Dabih"), ("Dabih", "DenebAlgedi")],
    },
    "Corvus": {
        "stars": {
            "GienahCorvi": (12.264, -17.542), "Algorab": (12.500, -16.515),
            "Kraz": (12.571, -23.397), "Minkar": (12.166, -22.622),
        },
        "lines": [("GienahCorvi", "Algorab"), ("Algorab", "Kraz"),
                  ("Kraz", "Minkar"), ("Minkar", "GienahCorvi")],
    },
    "Cancer": {
        "stars": {
            "Acubens": (8.975, 11.858), "AlTarf": (8.275, 9.186),
            "AsellusBorealis": (8.727, 21.468), "AsellusAustralis": (8.744, 18.154),
        },
        "lines": [("AlTarf", "AsellusAustralis"), ("AsellusAustralis", "AsellusBorealis"),
                  ("AsellusAustralis", "Acubens")],
    },
    "Canis Minor": {
        "stars": {
            "Procyon": (7.655, 5.225), "Gomeisa": (7.452, 8.289),
        },
        "lines": [("Procyon", "Gomeisa")],
    },
    "Virgo": {
        "stars": {
            "Spica": (13.420, -11.161), "Vindemiatrix": (13.036, 10.959),
            "Porrima": (12.694, -1.449), "Zaniah": (12.267, -0.667),
            "Heze": (13.582, -0.596),
        },
        "lines": [("Zaniah", "Porrima"), ("Porrima", "Heze"), ("Heze", "Spica"),
                  ("Porrima", "Vindemiatrix")],
    },
    "Libra": {
        "stars": {
            "Zubenelgenubi": (14.848, -16.042), "Zubeneschamali": (15.283, -9.383),
            "Brachium": (15.067, -25.282),
        },
        "lines": [("Zubenelgenubi", "Zubeneschamali"), ("Zubenelgenubi", "Brachium")],
    },
    "Pisces": {
        "stars": {
            "Alrescha": (2.036, 2.764), "EtaPsc": (1.525, 15.346),
        },
        "lines": [("Alrescha", "EtaPsc")],
    },
    "Aries": {
        "stars": {
            "Hamal": (2.119, 23.462), "Sheratan": (1.911, 20.808),
            "Mesarthim": (1.892, 19.294),
        },
        "lines": [("Hamal", "Sheratan"), ("Sheratan", "Mesarthim")],
    },
    "Cetus": {
        "stars": {
            "Menkar": (3.038, 4.090), "Diphda": (0.727, -17.987),
            "BatenKaitos": (1.771, -10.335),
        },
        "lines": [("Menkar", "BatenKaitos"), ("BatenKaitos", "Diphda")],
    },
    "Piscis Austrinus": {
        "stars": {
            "Fomalhaut": (22.961, -29.622), "EpsilonPsA": (22.906, -27.043),
        },
        "lines": [("Fomalhaut", "EpsilonPsA")],
    },
    "Crux": {
        "stars": {
            "Acrux": (12.443, -63.099), "Mimosa": (12.795, -59.689),
            "Gacrux": (12.519, -57.113), "Imai": (12.252, -58.749),
        },
        "lines": [("Gacrux", "Acrux"), ("Imai", "Mimosa")],
    },
    "Centaurus": {
        "stars": {
            "AlphaCen": (14.660, -60.834), "HadarCen": (14.064, -60.373),
        },
        "lines": [("AlphaCen", "HadarCen")],
    },
    "Hydra": {
        "stars": {
            "Alphard": (9.460, -8.659), "IotaHya": (9.664, -1.142),
        },
        "lines": [("Alphard", "IotaHya")],
    },
    "Octans": {
        "stars": {
            "NuOct": (21.699, -77.390), "BetaOct": (22.828, -81.383),
            "SigmaOct": (21.088, -88.957),
        },
        "lines": [("NuOct", "BetaOct"), ("BetaOct", "SigmaOct"), ("SigmaOct", "NuOct")],
    },
    "Triangulum Australe": {
        "stars": {
            "AlphaTrA": (16.811, -69.028), "BetaTrA": (15.919, -63.430),
            "GammaTrA": (16.233, -68.679),
        },
        "lines": [("AlphaTrA", "GammaTrA"), ("GammaTrA", "BetaTrA"), ("BetaTrA", "AlphaTrA")],
    },
    "Hydrus": {
        "stars": {
            "BetaHyi": (0.428, -77.254), "AlphaHyi": (1.984, -61.570),
            "GammaHyi": (3.848, -74.239),
        },
        "lines": [("BetaHyi", "AlphaHyi"), ("AlphaHyi", "GammaHyi")],
    },
}


def radec_to_vec(ra_hours, dec_deg):
    """Convert equatorial coordinates to a unit vector on the celestial sphere."""
    ra = math.radians(ra_hours * 15.0)
    dec = math.radians(dec_deg)
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    return np.array([x, y, z], dtype=np.float64)


# pre-compute the 3D vector for every star, once
for _data in CONSTELLATIONS.values():
    _data["vecs"] = {name: radec_to_vec(*coords) for name, coords in _data["stars"].items()}


# --------------------------------------------------------------------------
# 1a. Apparent magnitude — real relative brightness (lower/negative = brighter).
#     Rendering every star the same size is what makes a starfield look like
#     a wireframe model; real skies have a huge range of star brightness.
# --------------------------------------------------------------------------
MAGNITUDES = {
    "Dubhe": 1.79, "Merak": 2.37, "Phecda": 2.44, "Megrez": 3.32, "Alioth": 1.77,
    "Mizar": 2.23, "Alkaid": 1.86,
    "Polaris": 1.98, "Yildun": 4.35, "EpsUMi": 4.23, "ZetaUMi": 4.32, "Kochab": 2.08,
    "Pherkad": 3.05, "EtaUMi": 4.95,
    "Caph": 2.28, "Schedar": 2.24, "Navi": 2.47, "Ruchbah": 2.68, "Segin": 3.35,
    "Betelgeuse": 0.50, "Bellatrix": 1.64, "Mintaka": 2.23, "Alnilam": 1.69,
    "Alnitak": 1.74, "Saiph": 2.09, "Rigel": 0.13,
    "Deneb": 1.25, "Sadr": 2.23, "Gienah": 2.46, "DeltaCyg": 2.87, "Albireo": 3.05,
    "Vega": 0.03, "Sheliak": 3.52, "Sulafat": 3.25, "DeltaLyr": 4.30,
    "Regulus": 1.35, "Algieba": 2.08, "Zosma": 2.56, "Denebola": 2.14, "Chertan": 3.34,
    "Dschubba": 2.29, "Graffias": 2.56, "Antares": 0.96, "Sargas": 1.86,
    "Shaula": 1.62, "Lesath": 2.70,
    "Aldebaran": 0.85, "Elnath": 1.65, "Zeta_Tau": 3.00, "Hyadum": 3.65,
    "Castor": 1.58, "Pollux": 1.14, "Alhena": 1.93, "Wasat": 3.53, "Mebsuta": 3.06,
    "Sirius": -1.46, "Mirzam": 1.98, "Wezen": 1.83, "Adhara": 1.50,
    "Arcturus": -0.05, "Izar": 2.37, "Seginus": 3.03, "Nekkar": 3.49, "Muphrid": 2.68,
    "Alpheratz": 2.06, "Mirach": 2.05, "Almach": 2.10,
    "Capella": 0.08, "Menkalinan": 1.90, "Elnath2": 1.65, "Hassaleh": 2.69, "Almaaz": 2.92,
    "Markab": 2.48, "Scheat": 2.42, "Algenib": 2.83, "Alpheratz2": 2.06,
    "Mirfak": 1.79, "Algol": 2.12, "Atik": 2.85, "Miram": 3.76,
    "Alderamin": 2.44, "Alfirk": 3.23, "Errai": 3.21, "ZetaCep": 3.35,
    "Thuban": 3.65, "Rastaban": 2.79, "Eltanin": 2.23, "Altais": 3.07,
    "Kornephoros": 2.77, "ZetaHer": 2.81, "PiHer": 3.16, "EtaHer": 3.48,
    "Alphecca": 2.22, "Nusakan": 3.68, "ThetaCrB": 4.14, "GammaCrB": 3.84,
    "Altair": 0.76, "Tarazed": 2.72, "Alshain": 3.71,
    "KausAustralis": 1.85, "KausMedia": 2.72, "KausBorealis": 2.82,
    "Alnasl": 2.99, "Nunki": 2.05, "Ascella": 2.60,
    "Algedi": 3.57, "Dabih": 3.05, "DenebAlgedi": 2.85,
    "GienahCorvi": 2.59, "Algorab": 2.95, "Kraz": 2.65, "Minkar": 3.02,
    "Acubens": 4.25, "AlTarf": 3.53, "AsellusBorealis": 4.66, "AsellusAustralis": 3.94,
    "Procyon": 0.34, "Gomeisa": 2.89,
    "Spica": 0.97, "Vindemiatrix": 2.85, "Porrima": 2.74, "Zaniah": 3.89, "Heze": 3.38,
    "Zubenelgenubi": 2.75, "Zubeneschamali": 2.61, "Brachium": 2.65,
    "Alrescha": 3.82, "EtaPsc": 3.62,
    "Hamal": 2.00, "Sheratan": 2.64, "Mesarthim": 3.88,
    "Menkar": 2.54, "Diphda": 2.04, "BatenKaitos": 3.74,
    "Fomalhaut": 1.16, "EpsilonPsA": 4.17,
    "Acrux": 0.77, "Mimosa": 1.25, "Gacrux": 1.63, "Imai": 2.79,
    "AlphaCen": -0.27, "HadarCen": 0.61,
    "Alphard": 1.98, "IotaHya": 3.91,
    "NuOct": 3.76, "BetaOct": 4.15, "SigmaOct": 5.42,
    "AlphaTrA": 1.91, "BetaTrA": 2.85, "GammaTrA": 2.87,
    "BetaHyi": 2.80, "AlphaHyi": 2.86, "GammaHyi": 3.24,
}

# stars with a genuine warm (orange/red giant or supergiant) color in reality
WARM_STARS = {"Betelgeuse", "Antares", "Aldebaran", "Arcturus", "Mirach",
              "Pollux", "Gienah", "Hassaleh", "KausMedia",
              "Menkar", "Hamal", "Gacrux", "Vindemiatrix", "Alphard"}


def mag_to_scale(mag):
    """Apparent magnitude -> a rendering size/brightness multiplier.
    Real starlight falls off as ~10**(-0.4*mag); this uses a gentler curve
    so faint stars stay visible instead of disappearing entirely."""
    raw = 10 ** (-0.2 * (mag - 1.0))
    return max(0.35, min(1.7, raw))


def rotation_matrix(yaw, pitch):
    """Combined yaw (around vertical axis) + pitch (around horizontal axis)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rx @ ry


# --------------------------------------------------------------------------
# 1b. A faint, unnamed background star field — for atmosphere. Real dark
#     skies are dense with dim stars, not just the dozen bright named ones,
#     and this is most of what sells the "surreal planetarium" feel.
# --------------------------------------------------------------------------
def random_sphere_vec():
    """A uniformly random point on the unit sphere (Marsaglia's method)."""
    z = random.uniform(-1, 1)
    theta = random.uniform(0, 2 * math.pi)
    r = math.sqrt(1 - z * z)
    return np.array([r * math.cos(theta), r * math.sin(theta), z])


BACKGROUND_STARS = [random_sphere_vec() for _ in range(260)]


def name_seed(s):
    """A small stable number from a string, used to give each star its own
    twinkle rhythm without relying on Python's randomized hash()."""
    return sum(ord(c) for c in s) % 97


def twinkle(seed, t):
    """A smooth, per-star brightness oscillation between ~0.5 and 1.0."""
    phase = (seed * 0.647) % (2 * math.pi)
    freq = 0.6 + (seed % 7) * 0.11
    return 0.55 + 0.45 * math.sin(t * freq + phase)


def ema_point(prev, new, alpha):
    """Exponential moving average of a 2D point — smooths out the frame-to-
    frame noise that's inherent in any per-frame landmark detector, so a
    still hand doesn't translate into a shaky rotation/zoom."""
    if prev is None:
        return new
    return (prev[0] + (new[0] - prev[0]) * alpha,
            prev[1] + (new[1] - prev[1]) * alpha)


# --------------------------------------------------------------------------
# 2. Rendering — project the rotated sphere onto the screen
# --------------------------------------------------------------------------
BASE_ZOOM = 1.7      # "sphere radius" feel: bigger = more zoomed in, more of an
                     # immersive giant dome; smaller = see more sky at once,
                     # like looking at a small model from outside
MIN_ZOOM, MAX_ZOOM = 0.4, 4.0

LINE_COLOR  = (150, 120, 90)     # dim, cool slate — a guide, not a graphic
COOL_HALO   = (115, 100, 75)
COOL_CORE   = (235, 225, 205)
WARM_HALO   = (60, 95, 140)
WARM_CORE   = (150, 195, 255)


def draw_dome(img, yaw, pitch, t, zoom=BASE_ZOOM):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    f = min(w, h) * zoom
    R = rotation_matrix(yaw, pitch)

    def project(vec):
        rv = R @ vec
        if rv[1] <= 0.05:              # behind the viewer
            return None
        sx = cx + f * (rv[0] / rv[1])
        sy = cy - f * (rv[2] / rv[1])
        return (int(sx), int(sy))

    # faint unnamed background stars, twinkling, mostly tiny with a few bigger
    for i, vec in enumerate(BACKGROUND_STARS):
        p = project(vec)
        if p and 0 <= p[0] < w and 0 <= p[1] < h:
            b = twinkle(i, t)
            c = int(130 * b)
            r = 2 if i % 23 == 0 else 1
            cv2.circle(img, p, r, (c, c, c), -1, cv2.LINE_AA)

    # project every named star once, and draw the constellation lines as a
    # soft translucent pass (blended in, not drawn opaque) so they read as
    # guides rather than as the main graphic
    all_proj = {}
    line_layer = img.copy()
    for name, data in CONSTELLATIONS.items():
        proj = {}
        for star, vec in data["vecs"].items():
            p = project(vec)
            if p and -100 <= p[0] <= w + 100 and -100 <= p[1] <= h + 100:
                proj[star] = p
        for a, b in data["lines"]:
            if a in proj and b in proj:
                cv2.line(line_layer, proj[a], proj[b], LINE_COLOR, 1, cv2.LINE_AA)
        all_proj[name] = proj
    img[:] = cv2.addWeighted(img, 0.55, line_layer, 0.45, 0)

    # now the stars themselves, sized and colored by real apparent magnitude
    for name, proj in all_proj.items():
        for star, (sx, sy) in proj.items():
            scale = mag_to_scale(MAGNITUDES.get(star, 3.0))
            tw = 0.75 + 0.25 * twinkle(name_seed(name + star), t)
            warm = star in WARM_STARS
            halo, core = (WARM_HALO, WARM_CORE) if warm else (COOL_HALO, COOL_CORE)
            r_halo = max(2, int(9 * scale * tw))
            r_core = max(1, int(5 * scale * tw))
            cv2.circle(img, (sx, sy), r_halo, halo, -1, cv2.LINE_AA)
            cv2.circle(img, (sx, sy), r_core, core, -1, cv2.LINE_AA)
            cv2.circle(img, (sx, sy), 1, (255, 255, 255), -1, cv2.LINE_AA)

        if proj:
            label_pt = min(proj.values(), key=lambda p: p[1])
            lx, ly = label_pt[0] + 10, label_pt[1] - 12
            cv2.putText(img, name, (lx + 1, ly + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (20, 15, 20), 1, cv2.LINE_AA)      # soft shadow
            cv2.putText(img, name, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (215, 195, 225), 1, cv2.LINE_AA)   # muted label


def tint_cool(img, strength=0.40):
    """Nudge an image toward a cool night-sky blue, away from neutral gray."""
    out = img.astype(np.float32)
    out[:, :, 0] *= 1.0 + strength * 0.55   # more blue
    out[:, :, 1] *= 1.0 + strength * 0.05   # green ~unchanged
    out[:, :, 2] *= 1.0 - strength * 0.20   # a little less red
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_bloom(img, threshold=170, intensity=0.6):
    """Cheap glow: blur a small copy of the bright areas and add it back.
    Downsampling first keeps this fast enough for a live webcam feed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    bright = cv2.bitwise_and(img, img, mask=mask)
    small = cv2.resize(bright, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=6)
    glow = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 1.0, glow, intensity, 0)


_vignette_cache = {}

def apply_vignette(img, strength=0.55):
    """Darken the edges so the frame reads like looking up into a curved
    dome rather than a flat rectangle — cached per resolution since it's
    the same mask every frame."""
    h, w = img.shape[:2]
    key = (w, h, strength)
    if key not in _vignette_cache:
        yy, xx = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = math.hypot(cx, cy)
        vig = 1.0 - strength * (dist / max_dist) ** 2
        _vignette_cache[key] = np.clip(vig, 1.0 - strength, 1.0)[..., None]
    return np.clip(img.astype(np.float32) * _vignette_cache[key], 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 3. Hand tracking (MediaPipe Tasks API) — presence toggles the dome,
#    pinch + drag rotates it
# --------------------------------------------------------------------------
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

DIM_FACTOR = 0.20   # how much of the live room stays visible when a hand is up
                     # (0.0 = full blackout, 1.0 = no dimming at all)
PINCH_ENGAGE = 0.045         # normalized thumb-tip/index-tip distance to start a pinch
PINCH_RELEASE = 0.058        # ...and a larger one to end it (hysteresis, stops flicker
                              # right at the boundary from reading as pinch/no-pinch/pinch)
ROTATE_SPEED = -0.006
FADE_SPEED = 0.2           # per-frame ease toward the target (bigger = snappier fade)
POINT_SMOOTHING = 0.4        # EMA weight for raw fingertip positions (lower = smoother, laggier)
ZOOM_SMOOTHING = 0.25        # EMA weight for the zoom level itself

WINDOW_NAME = "Star Dome"
CAPTURE_WIDTH = 1280        # ask the camera for this size — bigger = bigger window,
CAPTURE_HEIGHT = 720        # but only as big as your camera actually supports


def ensure_model(retries=3, timeout=20):
    """Download the hand-landmark model once, next to this script.

    Uses a real browser User-Agent and a few retries — a bare urlretrieve()
    request has no headers at all, and some networks/CDNs (proxies,
    corporate firewalls, flaky Wi-Fi) will reset a connection that looks
    like a bot making a naked request.
    """
    if os.path.exists(MODEL_PATH):
        return

    headers = {"User-Agent": "Mozilla/5.0 (compatible; StarDomeSetup/1.0)"}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading hand_landmarker.task (attempt {attempt}/{retries})...")
            req = urllib.request.Request(MODEL_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp, \
                 open(MODEL_PATH, "wb") as out_file:
                out_file.write(resp.read())
            print("Done.")
            return
        except (urllib.error.URLError, http.client.RemoteDisconnected, TimeoutError) as e:
            last_err = e
            print(f"  attempt {attempt} failed: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)   # don't leave a partial/corrupt file behind
            time.sleep(1.5 * attempt)   # small backoff before retrying

    raise RuntimeError(
        "Could not download hand_landmarker.task automatically "
        f"(last error: {last_err}).\n"
        "This is almost always a network/firewall/VPN blocking the request, "
        "not a problem with the script.\n\n"
        "Fix: download the file yourself and place it next to star_dome.py:\n"
        f"  {MODEL_URL}\n"
        f"Save it as: {MODEL_PATH}\n"
        "Then just run the script again — it will skip the download."
    )


def make_landmarker(num_hands=2):
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def main():
    ensure_model()
    landmarker = make_landmarker(num_hands=2)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    fullscreen = False

    yaw, pitch = 0.0, 0.0
    zoom = BASE_ZOOM
    pinching = False
    prev_pt = None
    smooth_index = None       # EMA-smoothed fingertip positions, per mode
    smooth_thumb = None
    smooth_tip1 = None
    smooth_tip2 = None
    zoom_ref_dist = None      # None = not currently in a two-hand zoom gesture
    zoom_ref_zoom = None
    transition = 0.0        # 0 = plain camera feed, 1 = fully in the dome
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_now = time.time() - start_time
        result = landmarker.detect_for_video(mp_image, int(t_now * 1000))
        hands_list = result.hand_landmarks
        hand_present = bool(hands_list)

        # ease the fade level toward its target instead of cutting instantly
        target = 1.0 if hand_present else 0.0
        transition += (target - transition) * FADE_SPEED

        # build the "in the dome" layer: dimmed, cool-tinted, starry, glowing
        dome_layer = cv2.convertScaleAbs(frame, alpha=DIM_FACTOR, beta=0)
        dome_layer = tint_cool(dome_layer, strength=0.40)
        draw_dome(dome_layer, yaw, pitch, t_now, zoom=zoom)
        if transition > 0.05:           # skip the extra cost while fully faded out
            dome_layer = apply_bloom(dome_layer)
            dome_layer = apply_vignette(dome_layer)

        # cross-fade between the plain feed and the dome layer
        canvas = cv2.addWeighted(frame, 1 - transition, dome_layer, transition, 0)

        if len(hands_list) >= 2:
            # two hands -> zoom: spread apart to zoom in, bring together to
            # zoom out, same idea as a touchscreen pinch-zoom but two-handed
            smooth_index = smooth_thumb = None       # leaving one-hand mode
            tip1_raw = (hands_list[0][8].x * w, hands_list[0][8].y * h)
            tip2_raw = (hands_list[1][8].x * w, hands_list[1][8].y * h)
            smooth_tip1 = ema_point(smooth_tip1, tip1_raw, POINT_SMOOTHING)
            smooth_tip2 = ema_point(smooth_tip2, tip2_raw, POINT_SMOOTHING)
            dist = math.hypot(smooth_tip1[0] - smooth_tip2[0], smooth_tip1[1] - smooth_tip2[1])

            if zoom_ref_dist is None:
                zoom_ref_dist = dist
                zoom_ref_zoom = zoom
            else:
                target_zoom = zoom_ref_zoom * (dist / max(zoom_ref_dist, 1e-6))
                target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, target_zoom))
                zoom += (target_zoom - zoom) * ZOOM_SMOOTHING   # ease, don't snap

            pinching = False
            prev_pt = None       # don't let a stale rotate-drag jump on return to 1 hand

            p1 = (int(smooth_tip1[0]), int(smooth_tip1[1]))
            p2 = (int(smooth_tip2[0]), int(smooth_tip2[1]))
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.line(canvas, p1, p2, (140, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, p1, 8, (140, 220, 255), -1)
            cv2.circle(canvas, p2, 8, (140, 220, 255), -1)
            cv2.putText(canvas, "Zoom", (mid[0] - 20, mid[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 220, 255), 2, cv2.LINE_AA)

        elif len(hands_list) == 1:
            zoom_ref_dist = None      # left two-hand mode — clear the anchor
            smooth_tip1 = smooth_tip2 = None
            lm = hands_list[0]        # 21 landmarks for the one hand present
            thumb_tip, index_tip = lm[4], lm[8]
            dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
            # hysteresis: a different threshold to engage vs release a pinch,
            # so noise sitting right on one fixed boundary can't flicker it
            pinch_now = dist < PINCH_RELEASE if pinching else dist < PINCH_ENGAGE

            smooth_thumb = ema_point(smooth_thumb, (thumb_tip.x * w, thumb_tip.y * h), POINT_SMOOTHING)
            smooth_index = ema_point(smooth_index, (index_tip.x * w, index_tip.y * h), POINT_SMOOTHING)
            hand_pt = smooth_index

            if pinch_now:
                if pinching and prev_pt is not None:
                    yaw += (hand_pt[0] - prev_pt[0]) * ROTATE_SPEED
                    pitch += (hand_pt[1] - prev_pt[1]) * ROTATE_SPEED
                    # no clamp: pitch wraps all the way around, same as yaw —
                    # keep pinching and dragging and you'll rotate straight
                    # over a pole and come back around underneath
                prev_pt = hand_pt
                pinching = True
            else:
                pinching = False
                prev_pt = None

            pinch_color = (60, 220, 130) if pinching else (90, 90, 90)
            cv2.circle(canvas, (int(smooth_thumb[0]), int(smooth_thumb[1])), 6, pinch_color, -1)
            cv2.circle(canvas, (int(smooth_index[0]), int(smooth_index[1])), 6, pinch_color, -1)
            cv2.putText(canvas, "Pinch + drag to rotate  |  Two hands to zoom", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
        else:
            zoom_ref_dist = None
            smooth_index = smooth_thumb = smooth_tip1 = smooth_tip2 = None
            pinching = False
            prev_pt = None

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
