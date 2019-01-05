# -*- coding: utf-8 -*-
import pandas as pd
import scipy.io as scio
import geo_cal as geo
import numpy as np
from random import choice
import math
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, train_pt_path, signal_pair_path, src_path):
        self.train_pt_path = train_pt_path
        self.signal_pair_path = signal_pair_path
        self.src_path = src_path
        self.train_pts = None
        self.signal_pairs = None
        self.src = None
        self.salaries = None
        self.atcn = None
        self.price_mtx = None
        self.h_cap = None
        self.cell_height = None
        self.cell_width = None
        self.bnd = None
        self.theta = [1, 1, 1]
        self.J = math.inf
        self.signal_dis_list
        self.thresh = 0.001
        self.alfa = 1
        self.ipsilon = 0.001
        self.nrows = 0
        self.ncols = 0
        self.plt_pt = None

    def optimize(self):
        self.train_pts = pd.read_excel(self.train_pt_path)
        self.signal_pairs = pd.read_excel(self.signal_pair_path)
        self.signal_dis_list = self.signal_pairs['dis']
        self.src = scio.loadmat(self.src_path)
        self.salaries = self.src['salarys'][0]
        self.atcn = self.src['atcnMtx'][0]
        self.price_mtx = self.src['priceMtx'][0]
        self.h_cap = self.src['HLimit'][0]
        self.cell_height = self.src['cellH'][0]
        self.cell_width = self.src['cellW'][0]
        self.bnd = self.src['bnd'][0]
        self.nrows = self.src['nrows1km'][0]
        self.ncols = self.src['ncols1km'][0]

        # open the interactive mode
        plt.ion()
        plt.title("monitor")
        x = 0
        # self.scaling()
        while(self.J > self.thresh):
            self.J = self.match()
            if self.plt_pt is not None:
                plt.plot([self.plt_pt[0], x], [self.plt_pt[1], self.J])
            plt.scatter(x, self.J)
            plt.annotate(str(self.theta), xy=(x, self.J), xycoords='data',
                         xytext=(0, +0.1), textcoords='offset point')
            self.plt_pt = [x, self.J]
            x += 1
            if self.J > self.thresh:
                break
            else:
                for i in range(self.theta):
                    ipsilon_plus_theta = self.theta
                    ipsilon_minus_theta = self.theta
                    ipsilon_plus_theta[i] += self.ipsilon
                    ipsilon_minus_theta[i] -= self.ipsilon
                    prox_deriv = (self.match(ipsilon_plus_theta) -
                                  self.match(ipsilon_minus_theta)) / (2 * self.ipsilon)
                    self.theta[i] -= (self.alfa * prox_deriv)
        plt.ioff()
        plt.show()

    def match(self, theta):
        train_dis_list = []
        for i in range(self.trian_pts.shape[0]):
            train_pt = self.trian_pts.ioc[i]
            salary = self.salaries[i]
            dis_mtx = self.get_dis_mtx(train_pt)
            field = self.atcn - \
                (theta[0] * self.price_mtx + theta[1]
                 * dis_mtx**theta[2]) / salary + self.h_cap
            location = np.where(field == np.max(field))
            h_row_idx = choice(location[0])
            h_col_idx = choice(location[1])
            self.h_cap[h_row_idx][h_col_idx] -= 1
            if self.h_cap[h_row_idx][h_col_idx] <= 0:
                self.h_cap[h_row_idx][h_col_idx] = -math.inf
            h_lng_lat = geo.GetLngLat(
                h_row_idx, h_col_idx, self.cell_height, self.cell_width, self.bnd)
            dis = geo.Manhattan(
                train_pt['lng'], train_pt['lat'], h_lng_lat[0], h_lng_lat[1])
            train_dis_list.append(dis)
        J = self.get_j(train_dis_list, self.signal_dis_list)
        return J

    # def scaling(self):
    #     a = 1
    #     print(a)

    def get_dis_mtx(self, pt):
        dis_mtx = np.zeros((self.nrows, self.ncols))
        for i in range(self.nrows):
            for j in range(self.ncols):
                lng_lat = geo.GetLngLat(
                    i, j, self.cell_height, self.cell_height, self.bnd)
                dis_mtx[i][j] = geo.Manhattan(
                    pt['lng'], pt['lat'], lng_lat[0], lng_lat[1])
        return dis_mtx

    def get_j(train_dis_list, signal_dis_list):
        train_dis_list = round(train_dis_list)
        signal_dis_list = round(signal_dis_list)
        max_dis = np.max([np.max(train_dis_list), np.max(signal_dis_list)])
        train_hist = np.zeros((max_dis, 1))
        signal_hist = np.zeros((max_dis, 1))
        for i in range(len(train_dis_list)):
            train_hist[train_dis_list[i] + 1] += 1
        for i in range(len(signal_dis_list)):
            signal_hist[signal_dis_list[i] + 1] += 1
        train_hist /= len(train_dis_list)
        signal_hist /= len(signal_dis_list)
        dis_hist_coef = np.corrcoef(train_hist, signal_hist)[0][1]
        return 1 - dis_hist_coef


if __name__ == '__main__':
    # o = Optimizer('test.xls', '', '')
    # o.optimize()
    # print('a')
    # a = pd.read_excel('test.xls')
    # a = scio.loadmat('data/src1215_1.mat')
    # b = a['atcnMtx'][0]
    a = pd.read_csv('data/pairs_from_clean_1_300.csv')
    print(a)
