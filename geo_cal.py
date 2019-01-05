# -*- coding: utf-8 -*-
import math


# GETCOOR 得到行列数对应的网格中心点的坐标
def GetLngLat(row, col, cellH, cellW, bnd):
    lng = (col - 0.5) * cellW + bnd[0]
    lat = bnd[1] - (row - 0.5) * cellH
    return [lng, lat]


def Getxy(lng, lat, cellH, cellW, bnd):
    row_idx = math.ceil((lng - bnd[0]) / cellW)
    col_idx = (bnd[1] - bnd[3]) / cellH - math.ceil((lat - bnd[3]) / cellH) + 1
    return [row_idx, col_idx]


def Manhattan(lng1, lat1, lng2, lat2):
    # 计算两点之间的曼哈顿距离
    lmd1 = ang2rad(lng1)
    lmd2 = ang2rad(lng2)
    fai1 = ang2rad(lat1)
    fai2 = ang2rad(lat2)
    manDis = 6378.137 * (abs(fai1 - fai2)) + 6378.137 * \
        math.acos((fai1 + fai2) / 2) * abs(lmd1 - lmd2)
    return manDis


def ang2rad(angle):
    return angle * math.pi / 180
