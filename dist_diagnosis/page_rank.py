import copy

class WPR3Node:
    def __init__(self, weight):  # weight:[w12, w13, w23]
        self.weight = weight
        self.p = [0.15] * 3
        self.d = 0.85

    def iter(self):
        # p[0]
        tmp = self.p[1] * self.weight[0] * self.weight[0] + self.p[2] * self.weight[1] * self.weight[1]
        self.p[0] = (1 - self.d) + self.d * tmp

        # p[1]
        tmp = self.p[0] * self.weight[0] * self.weight[0] + self.p[2] * self.weight[2] * self.weight[2]
        self.p[1] = (1 - self.d) + self.d * tmp

        # p[2]
        tmp = self.p[0] * self.weight[1] * self.weight[1] + self.p[1] * self.weight[2] * self.weight[2]
        self.p[2] = (1 - self.d) + self.d * tmp

    def page_rank(self, num):
        for _ in range(num):
            self.iter()
        return self.p


class WPR4Node:
    def __init__(self, weight):  # weight:[w01, w02, w03, w12, w13, w23]
        self.weight = weight
        self.p = [0.15] * 4
        self.d = 0.85

    def iter(self):
        # p[0]
        tmp = (self.p[1] * self.weight[0] * self.weight[0]
               + self.p[2] * self.weight[1] * self.weight[1]
               + self.p[3] * self.weight[2] * self.weight[2])
        self.p[0] = (1 - self.d) + self.d * tmp

        # p[1]
        tmp = (self.p[0] * self.weight[0] * self.weight[0]
               + self.p[2] * self.weight[3] * self.weight[3]
               + self.p[3] * self.weight[4] * self.weight[4])
        self.p[1] = (1 - self.d) + self.d * tmp

        # p[2]
        tmp = (self.p[0] * self.weight[1] * self.weight[1]
               + self.p[1] * self.weight[3] * self.weight[3]
               + self.p[3] * self.weight[5] * self.weight[5])
        self.p[2] = (1 - self.d) + self.d * tmp

        # p[3]
        tmp = (self.p[0] * self.weight[2] * self.weight[2]
               + self.p[1] * self.weight[4] * self.weight[4]
               + self.p[2] * self.weight[5] * self.weight[5])
        self.p[3] = (1 - self.d) + self.d * tmp

    def page_rank(self, num):
        for _ in range(num):
            self.iter()
        return self.p


class WPRNNode():
    def __init__(self, weight, node_num):
        self.p = [0.15] * node_num
        self.d = 0.85
        if node_num == 0:
            raise NotImplementedError
        assert len(weight) == node_num
        assert len(weight[0]) == node_num
        self.w = weight

    def iter(self):
        new_p = copy.copy(self.p)
        for i in range(len(self.p)):
            tmp = 0
            for j in range(len(self.p)):
                if j != i:
                    tmp += self.p[j] * self.w[i][j] * self.w[i][j]
            new_p[i] = (1 - self.d) + self.d * tmp

        self.p = new_p

    def page_rank(self, num):
        for _ in range(num):
            self.iter()
        return self.p


if __name__ == '__main__':
    p = WPR3Node([0.8, 0.9, 0.5])
    print(p.page_rank(30))
    print(p.page_rank(10))
    print(p.page_rank(10000))
