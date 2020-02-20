import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import random
import os
import matplotlib


def action_to_quiver(a):
    vt = a[0]
    theta = a[1]
    vtx = vt * math.cos(theta)
    vty = vt * math.sin(theta)
    return vtx, vty


def permute_Y(xs, ys):
    P = np.zeros((len(ys), len(xs)))
    j = len(xs)
    for i in range(len(ys)):
        j = j - 1
        P[i, j] = 1
    return P


def my_meshgrid(x, y):
    x = list(x)
    xm = []
    for i in range(len(x)):
        xm.append(x)
    xm = np.asarray(xm)

    y = list(y)
    y.reverse()
    ym = []
    for i in range(len(y)):
        ym.append(y)
    ym = np.asarray(ym)

    return xm, ym.T


def plot_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y, fname=None, lastfig=None):
    # time calculation and state trajectory
    trajectory = []
    xtr = []
    ytr = []
    vtx_list = []
    vty_list = []

    t, i, j = g.start_state
    # print(t,x,y,vStream_x[t,x,y])

    g.set_state((t, i, j))
    # print(g.current_state())
    trajectory.append((i, j))
    a = policy[g.current_state()]
    vtx, vty = action_to_quiver(a)
    vtx_list.append(vtx)
    vty_list.append(vty)

    xtr.append(xs[j])
    ytr.append(ys[g.ni - 1 - i])

    # set grid
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    minor_xticks = np.arange(xs[0] - 0.5 * g.dj, xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(ys[0] - 0.5 * g.di, ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(xs[0], xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(ys[0], ys[-1] + 2 * g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.grid(which='major', color='#CCCCCC', linestyle='')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')

    plt.quiver(xtr, ytr, vtx_list, vty_list)

    plt.plot(xtr, ytr, label='Agent\'s Path')
    plt.scatter(xtr, ytr, label='Visited States')

    plt.title("Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")

    # plots start point
    st_point = trajectory[0]

    plt.scatter(xs[st_point[1]], ys[g.ni - 1 - st_point[0]], c='g', label='Start Location')
    # plots current point
    # plt.scatter(xs[trajectory[-1][0]], ys[g.ni-1-trajectory[-1][1]], c='k')
    # plots end point
    plt.scatter(xs[g.endpos[1]], ys[g.ni - 1 - g.endpos[0]], c='r', label='Target Location')
    # plots current point
    plt.scatter(xs[j], ys[g.ni - 1 - i])
    color_matrix=np.sqrt(vStream_x[t, :, :]**2 + vStream_y[t, :, :]**2)
    plt.quiver(X, Y, vStream_x[t, :, :], vStream_y[t, :, :], color_matrix, alpha=0.8)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(loc='upper right')
    # plt.show()
    if fname != None and lastfig==None:
        filename = fname + str(t) + ".png"
        plt.savefig(filename)
        plt.close()

    # print("in loop---")
    G = 0
    flag=False

    while not g.is_terminal() :

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        minor_xticks = np.arange(xs[0] - 0.5 * g.dj, xs[-1] +  2*2*g.dj, g.dj)
        minor_yticks = np.arange(ys[0] - 0.5 * g.di, ys[-1] +  2*g.di, g.di)

        major_xticks = np.arange(xs[0], xs[-1] + 2*g.dj, 5 * g.dj)
        major_yticks = np.arange(ys[0], ys[-1] + 2*g.di, 5 * g.di)

        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(minor_yticks, minor=True)
        ax.set_xticks(major_xticks)
        ax.set_yticks(major_yticks)

        ax.grid(which='major', color='#CCCCCC', linestyle='')
        ax.grid(which='minor', color='#CCCCCC', linestyle='--')

        print("state", (t, i, j))
        print("action", a)
        # print("vfx", vStream_x[t, i, j])
        # print("vfy", vStream_y[t, i, j])

        r = g.move( a, vStream_x[t, i, j], vStream_y[t, i, j] )
        G = G + r
        (t, i, j) = g.current_state()

        # print(g.current_state(), (t, x, y), a, vStream_x[t, x, y], vStream_y[t, x, y])
        trajectory.append((i, j))

        xtr.append(xs[j])
        ytr.append(ys[g.ni - 1 - i])


        if not g.is_terminal() :
            a = policy[g.current_state()]
        else:
            a=None
            flag=True

        if a != None:
            vtx, vty = action_to_quiver(a)
            vtx_list.append(vtx)
            vty_list.append(vty)
            plt.quiver(xtr, ytr, vtx_list, vty_list)
        else:
            plt.quiver(xtr[0:len(xtr) - 1], ytr[0:len(ytr) - 1], vtx_list, vty_list)

        plt.plot(xtr, ytr, label='Agent\'s Path')
        plt.scatter(xtr, ytr, label='Visited States')

        plt.scatter(xs[st_point[1]], ys[g.ni - 1 - st_point[0]], c='g', label='Start Location')
        # plt.scatter(xs[trajectory[-1][0]], ys[g.ni - 1 - trajectory[-1][1]], c='k')
        plt.scatter(xs[g.endpos[1]], ys[g.ni - 1 - g.endpos[0]], c='r', label='Target Location')
        plt.scatter(xs[j], ys[g.ni - 1 - i])

        color_matrix = np.sqrt(vStream_x[t, :, :] ** 2 + vStream_y[t, :, :] ** 2)
        plt.quiver(X, Y, vStream_x[t, :, :], vStream_y[t, :, :], color_matrix, alpha=0.6)

        plt.title("Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='upper right')

        plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

        if fname!=None:
            if lastfig==True and flag==True:
                filename = fname + str(t) + ".png"
                plt.savefig(filename)
                plt.close()
            elif lastfig==None:
                filename = fname + str(t) + ".png"
                plt.savefig(filename)
                plt.close()


    return trajectory, (t, i, j), G


def plot_exact_trajectory(g, policy, xs, ys, X, Y, vStream_x, vStream_y,fname=None, lastfig=None):
    # time calculation and state trajectory
    trajectory = []
    xtr = []
    ytr = []
    vtx_list = []
    vty_list = []

    t, i, j = g.start_state
    # print(t,x,y,vStream_x[t,x,y])

    g.set_state((t, i, j))
    # print(g.current_state())
    trajectory.append((i, j))
    a = policy[g.current_state()]
    vtx, vty = action_to_quiver(a)
    vtx_list.append(vtx)
    vty_list.append(vty)

    xtr.append(g.x)
    ytr.append(g.y)

    #set grid
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    minor_xticks = np.arange(xs[0] - 0.5 * g.dj, xs[-1] + 2*g.dj, g.dj)
    minor_yticks = np.arange(ys[0] - 0.5 * g.di, ys[-1] + 2*g.di, g.di)

    major_xticks = np.arange(xs[0], xs[-1] + 2*g.dj, 5 * g.dj)
    major_yticks = np.arange(ys[0], ys[-1] + 2*g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.grid(which='major', color='#CCCCCC', linestyle='')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')

    plt.quiver(xtr, ytr, vtx_list, vty_list)


    plt.plot(xtr, ytr, label='Agent\'s Path')
    plt.scatter(xtr, ytr, label='Visited States')

    plt.title("Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc='upper right')

    # plots start point
    st_point = trajectory[0]
    plt.scatter(xs[st_point[1]], ys[g.ni - 1 - st_point[0]], c='g',label='Start Location')
    # plots current point
    # plt.scatter(xs[trajectory[-1][0]], ys[g.ni-1-trajectory[-1][1]], c='k')
    # plots end point
    plt.scatter(xs[g.endpos[1]], ys[g.ni - 1 - g.endpos[0]], c='r',label='Target Location')
    # plots current point
    plt.scatter(g.x, g.y)

    color_matrix = np.sqrt(vStream_x[t, :, :] ** 2 + vStream_y[t, :, :] ** 2)
    plt.quiver(X, Y, vStream_x[t, :, :], vStream_y[t, :, :], color_matrix, alpha=0.8)

    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    if fname != None and lastfig == None:
        filename = fname + str(t) + ".png"
        plt.savefig(filename)
        plt.close()

    G = 0
    flag=False

    while not g.is_terminal():

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1)
        minor_xticks = np.arange(xs[0] - 0.5 * g.dj, xs[-1] +  2*g.dj, g.dj)
        minor_yticks = np.arange(ys[0] - 0.5 * g.di, ys[-1] +  2*g.di, g.di)

        major_xticks = np.arange(xs[0], xs[-1] + 2*g.dj , 5 * g.dj)
        major_yticks = np.arange(ys[0], ys[-1] + 2*g.di , 5 * g.di)

        ax.set_xticks(minor_xticks, minor=True)
        ax.set_yticks(minor_yticks, minor=True)
        ax.set_xticks(major_xticks)
        ax.set_yticks(major_yticks)

        ax.grid(which='major', color='#CCCCCC', linestyle='')
        ax.grid(which='minor', color='#CCCCCC', linestyle='--')

        # print("state", (t, i, j))
        # print("action", a)
        # print("vfx", vStream_x[t, i, j])
        # print("vfy", vStream_y[t, i, j])

        r = g.move_exact(a, vStream_x[t, i, j], vStream_y[t, i, j])
        G = G + r
        (t, i, j) = g.current_state()

        # print(g.current_state(), (t, x, y), a, vStream_x[t, x, y], vStream_y[t, x, y])
        trajectory.append((i, j))

        xtr.append(g.x)
        ytr.append(g.y)

        if not g.is_terminal():
            a = policy[g.current_state()]
        else:
            a = None
            flag = True

        if a != None:
            vtx, vty = action_to_quiver(a)
            vtx_list.append(vtx)
            vty_list.append(vty)
            plt.quiver(xtr, ytr, vtx_list, vty_list)
        else:
            plt.quiver(xtr[0:len(xtr) - 1], ytr[0:len(ytr) - 1], vtx_list, vty_list)

        plt.plot(xtr, ytr, label='Agent\'s Path')
        plt.scatter(xtr, ytr, label='Visited States')

        plt.scatter(xs[st_point[1]], ys[g.ni - 1 - st_point[0]], c='g', label='Start Location')
        # plt.scatter(xs[trajectory[-1][0]], ys[g.ni - 1 - trajectory[-1][1]], c='k')
        plt.scatter(xs[g.endpos[1]], ys[g.ni - 1 - g.endpos[0]], c='r',label='Target Location')
        plt.scatter(g.x, g.y)

        color_matrix = np.sqrt(vStream_x[t, :, :] ** 2 + vStream_y[t, :, :] ** 2)
        plt.quiver(X, Y, vStream_x[t, :, :], vStream_y[t, :, :], color_matrix, alpha=0.8)
        plt.grid()

        plt.title("Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='upper right')

        plt.gca().set_aspect('equal', adjustable='box')


        # plt.show()
        if fname != None:
            if lastfig == True and flag == True:
                filename = fname + str(t) + ".png"
                plt.savefig(filename)
                plt.close()
            elif lastfig == None:
                filename = fname + str(t) + ".png"
                plt.savefig(filename)
                plt.close()

    return trajectory, (t, i, j), G



def plot_value_funtion(fname):
    f = open(fname)
    V = {}
    V = eval(f.read())

    state_mat =[]
    val_arr =[]
    for Keys in V.keys():
        state_mat.append([Keys[0], Keys[1], Keys[2]])
        val_arr.append(V[Keys])

    X = np.asarray(state_mat)
    Z = np.asarray(val_arr)








def calculate_reward_const_dt(dt, xs, ys, so, sn, Vx, Vy):
    if (Vx == 0 and Vy == 0) or (so == sn):
        dt_new = dt
    else:
        ni = len(ys)

        io = so[1]
        jo = so[2]
        xo = xs[jo]
        yo = ys[ni - 1 - io]

        inw = sn[1]
        jnw = sn[2]
        xnw = xs[jnw]
        ynw = ys[ni - 1 - inw]

        h = ((xnw - xo) ** 2 + (ynw - yo) ** 2) ** 0.5

        dt_new= h/((Vx ** 2 + Vy ** 2) ** 0.5)

    return -dt_new


def calculate_reward_var_dt(dt, xs, ys, so, sn, xp, yp, Vx, Vy):
    if (Vx == 0 and Vy == 0) or (so == sn):
        dt_new = dt
    else:
        ni = len(ys)

        io = so[1]
        jo = so[2]
        xo = xs[jo]
        yo = ys[ni - 1 - io]

        inw = sn[1]
        jnw = sn[2]
        xnw = xs[jnw]
        ynw = ys[ni - 1 - inw]

        # print((xo, yo), (xp, yp), (xnw, ynw))
        A = B = C = 0
        # distance of sn (new state) from line
        if xnw - xo == 0:
            d = xp - xnw
        else:
            # line Ax+ By + C =0 line along Vnet
            A = (yp - yo) / (xp - xo)
            B = -1
            C = yo - (xo * (yp - yo) / (xp - xo))

            d = (A * xnw + B * ynw + C) / (A ** 2 + B ** 2) ** 0.5

        # distance from so to sn
        h = ((xnw - xo) ** 2 + (ynw - yo) ** 2) ** 0.5

        # print("d", d, "h", h)

        # length of tangent

        tan_len = (h ** 2 - d ** 2) ** 0.5
        if h ** 2 < d ** 2:
            print("----------------------BUG HAI ------------------------------")
            print("so,sn=", so, sn)
            print("(xo,yo)=", (xo, yo))
            print("(xnw,ynw)=", (xnw, ynw))
            print("(xp,yp=", (xp, yp))
            print("A, B, C =", A, B, C)
            print("h=", h)
            print("d=", d)
            print("tan_len=", tan_len)
        # actual distance travelled in dT
        tot_dis = ((xp - xo) ** 2 + (yp - yo) ** 2) ** 0.5
        # print("tan_len=", tan_len, "tot_dis", tot_dis)

        err = tot_dis - tan_len
        # print(err)

        dt_new = dt - (err / ((Vx ** 2 + Vy ** 2) ** 0.5))

    return -dt_new


"""

The functions below are used in the stochatic version of the DP 
Problem

"""


# state transition probability derived from prob dist of velocities
def p_sas(xi, xf, yi, yf, sigx, sigy):
    def pdf(x, y):
        ux = 0
        uy = 0
        return np.exp(- ((((x - ux) ** 2) / (2 * (sigx ** 2))) + (((y - uy) ** 2) / (2 * (sigy ** 2))))) / (
                2 * math.pi * sigx * sigy)

    return integrate.nquad(pdf, [[xi, xf], [yi, yf]])[0]


# generates matrix with samples from normal curve. Used to simulate uncertainty in velocity field
def gaussian_matrix(t, r, c, u, sigma):
    a = np.zeros((t, r, c))
    for k in range(t):
        for i in range(r):
            for j in range(c):
                a[k, i, j] = np.random.normal(u, sigma)
    return a


# finds number of states s' on either sides where p_sas is >thresh , i.e. significant
def find_mn(dx, dy, sigx, sigy, thresh):
    i = 0
    j = 0
    val = 1
    while val > thresh:
        val = p_sas((j - 0.5) * dx, (j + 0.5) * dx, (i - 0.5) * dy, (i + 0.5) * dy, sigx, sigy)
        j = j + 1
    m = j - 1

    val = 1
    i = 0
    j = 0
    while val > thresh:
        val = p_sas((j - 0.5) * dx, (j + 0.5) * dx, (i - 0.5) * dy, (i + 0.5) * dy, sigx, sigy)
        i = i + 1
    n = i - 1
    return m, n


def transition_probs(m, n, dx, dy, sigx, sigy, thresh):
    psas_dict = {}
    for i in range(-n, n + 1, 1):
        for j in range(-m, m + 1, 1):
            prob = p_sas((j - 0.5) * dx, (j + 0.5) * dx, (i - 0.5) * dy, (i + 0.5) * dy, sigx, sigy)
            if prob > thresh:
                psas_dict[(i, j)] = prob
    return psas_dict, psas_dict.keys()


def stoch_vel_field(d_vStream, xm, ym, cov_sigma, coef):
    t, r, c = d_vStream.shape
    st_vStream=np.zeros((t,r,c))
    fxgrd=np.ndarray.flatten(xm)
    fygrd=np.ndarray.flatten((ym))
    for k in range(t):
        fl_vel=np.ndarray.flatten(d_vStream[k,:,:])
        l=len(fl_vel)
        cov=np.zeros((l,l))
        for i in range(l):
            for j in range(l):
                rsqm=(fxgrd[i]-fxgrd[j])**2 + (fygrd[i]-fygrd[j])**2
                cov[i,j]=coef*np.exp(-rsqm/(2*cov_sigma**2))
        fl_st_vel=np.random.multivariate_normal(fl_vel,cov)
        st_vStream[k,:,:]=fl_st_vel.reshape((r,c))
    return st_vStream





def stochastic_action(policy, s, g, eps):
    p = np.random.random()
    n = len(g.actions[s])
    if p < (1 - eps + (eps / n)):
        newa = policy[s]
    else:
        i = random.randint(0, n - 1)
        newa = g.actions[s][i]
    return newa


def random_action(a, s, g, eps):
    p = np.random.random()
    n = len(g.actions[s])
    if p < (1 - eps + (eps / n)):
        newa = a
    else:
        i = random.randint(0, n - 1)
        newa = g.actions[s][i]

    return newa

def initialise_policy(g, action_states):
    policy = {}

    Pi = math.pi
    for s in action_states:
        t, i, j = s
        # print("start", s)
        i2, j2 = g.endpos
        # print("i2,j2", i2, j2)
        if j2 == j:
            if i2 > i:
                policy[s] = (1, 1.5 * Pi)
            elif i2 < i:
                policy[s] = (1, 0.5 * Pi)
        elif j2 > j:
            if i2 > i:
                policy[s] = (1, 1.75 * Pi)
            elif i2 < i:
                policy[s] = (1, 0.25 * Pi)
            elif i2 == i:
                policy[s] = (1, 0)
        elif j2 < j:
            if i2 > i:
                policy[s] = (1, 1.25 * Pi)
            elif i2 < i:
                policy[s] = (1, 0.75 * Pi)
            elif i2 == i:
                policy[s] = (1, Pi)
        # if i == 9 or j == 9:
        #     print("state, action ", s, policy[s])

    return policy


def initialise_Q_N(action_states, g):
    Q = {}
    N = {}
    for s in action_states:
        Q[s] = {}
        N[s] = {}
        for a in g.actions[s]:
            Q[s][a] = 0
            N[s][a] = 0

    return Q, N

def initialise_Q_N1(action_states, g):
    Q = {}
    N = {}
    for s in action_states:
        Q[s] = {}
        N[s] = {}
        for a in g.actions[s]:
            Q[s][a] = 0
            N[s][a] = 1

    return Q, N


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        # print("dict test:", k , v)
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

def writePolicytoFile(policy):
    outputfile = open('output.txt', 'w')
    print(policy, file=outputfile)
    outputfile.close()
    return




def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


