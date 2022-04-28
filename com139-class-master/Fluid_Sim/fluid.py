"""
Based on the Jos Stam paper https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games
and the mike ash vulgarization https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

https://github.com/Guilouf/python_realtime_fluidsim
"""
import random
import numpy as np
import math

class Fluid:

    def __init__(self):
        self.rotx = 1
        self.roty = 1
        self.cntx = 1
        self.cnty = -1

        self.size = 60  # map size
        self.dt = 0.2  # time interval
        self.iter = 2  # linear equation solving iteration number

        self.diff = 0.0000  # Diffusion
        self.visc = 0.0000  # viscosity

        self.s = np.full((self.size, self.size), 0, dtype=float)        # Previous density
        self.density = np.full((self.size, self.size), 0, dtype=float)  # Current density

        # array of 2d vectors, [x, y]
        self.velo = np.full((self.size, self.size, 2), 0, dtype=float)
        self.velo0 = np.full((self.size, self.size, 2), 0, dtype=float)

    def step(self):
        self.diffuse(self.velo0, self.velo, self.visc)

        # x0, y0, x, y
        self.project(self.velo0[:, :, 0], self.velo0[:, :, 1], self.velo[:, :, 0], self.velo[:, :, 1])

        self.advect(self.velo[:, :, 0], self.velo0[:, :, 0], self.velo0)
        self.advect(self.velo[:, :, 1], self.velo0[:, :, 1], self.velo0)

        self.project(self.velo[:, :, 0], self.velo[:, :, 1], self.velo0[:, :, 0], self.velo0[:, :, 1])

        self.diffuse(self.s, self.density, self.diff)

        self.advect(self.density, self.s, self.velo)

    def lin_solve(self, x, x0, a, c):
        """Implementation of the Gauss-Seidel relaxation"""
        c_recip = 1 / c

        for iteration in range(0, self.iter):
            # Calculates the interactions with the 4 closest neighbors
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])) * c_recip

            self.set_boundaries(x)

    def set_boundaries(self, table):
        """
        Boundaries handling
        :return:
        """

        if len(table.shape) > 2:  # 3d velocity vector array
            # Simulating the bouncing effect of the velocity array
            # vertical, invert if y vector
            table[:, 0, 1] = - table[:, 0, 1]
            table[:, self.size - 1, 1] = - table[:, self.size - 1, 1]

            # horizontal, invert if x vector
            table[0, :, 0] = - table[0, :, 0]
            table[self.size - 1, :, 0] = - table[self.size - 1, :, 0]

        table[0, 0] = 0.5 * (table[1, 0] + table[0, 1])
        table[0, self.size - 1] = 0.5 * (table[1, self.size - 1] + table[0, self.size - 2])
        table[self.size - 1, 0] = 0.5 * (table[self.size - 2, 0] + table[self.size - 1, 1])
        table[self.size - 1, self.size - 1] = 0.5 * table[self.size - 2, self.size - 1] + \
                                              table[self.size - 1, self.size - 2]

    def diffuse(self, x, x0, diff):
        if diff != 0:
            a = self.dt * diff * (self.size - 2) * (self.size - 2)
            self.lin_solve(x, x0, a, 1 + 6 * a)
        else:  # equivalent to lin_solve with a = 0
            x[:, :] = x0[:, :]

    def project(self, velo_x, velo_y, p, div):
        # numpy equivalent to this in a for loop:
        # div[i, j] = -0.5 * (velo_x[i + 1, j] - velo_x[i - 1, j] + velo_y[i, j + 1] - velo_y[i, j - 1]) / self.size
        div[1:-1, 1:-1] = -0.5 * (
                velo_x[2:, 1:-1] - velo_x[:-2, 1:-1] +
                velo_y[1:-1, 2:] - velo_y[1:-1, :-2]) / self.size
        p[:, :] = 0

        self.set_boundaries(div)
        self.set_boundaries(p)
        self.lin_solve(p, div, 1, 6)

        velo_x[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.size
        velo_y[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.size

        self.set_boundaries(self.velo)

    def advect(self, d, d0, velocity):
        dtx = self.dt * (self.size - 2)
        dty = self.dt * (self.size - 2)

        for j in range(1, self.size - 1):
            for i in range(1, self.size - 1):
                tmp1 = dtx * velocity[i, j, 0]
                tmp2 = dty * velocity[i, j, 1]
                x = i - tmp1
                y = j - tmp2

                if x < 0.5:
                    x = 0.5
                if x > (self.size - 1) - 0.5:
                    x = (self.size - 1) - 0.5
                i0 = math.floor(x)
                i1 = i0 + 1.0

                if y < 0.5:
                    y = 0.5
                if y > (self.size - 1) - 0.5:
                    y = (self.size - 1) - 0.5
                j0 = math.floor(y)
                j1 = j0 + 1.0

                s1 = x - i0
                s0 = 1.0 - s1
                t1 = y - j0
                t0 = 1.0 - t1

                i0i = int(i0)
                i1i = int(i1)
                j0i = int(j0)
                j1i = int(j1)

                try:
                    d[i, j] = s0 * (t0 * d0[i0i, j0i] + t1 * d0[i0i, j1i]) + \
                              s1 * (t0 * d0[i1i, j0i] + t1 * d0[i1i, j1i])
                except IndexError:
                    # tmp = str("inline: i0: %d, j0: %d, i1: %d, j1: %d" % (i0, j0, i1, j1))
                    # print("tmp: %s\ntmp1: %s" %(tmp, tmp1))
                    raise IndexError
        self.set_boundaries(d)

    def turn(self):
        self.cntx += 1
        self.cnty += 1
        if self.cntx == 3:
            self.cntx = -1
            self.rotx = 0
        elif self.cntx == 0:
            self.rotx = self.roty * -1
        if self.cnty == 3:
            self.cnty = -1
            self.roty = 0
        elif self.cnty == 0:
            self.roty = self.rotx
        return self.rotx, self.roty


def velType(frame, vX, vY, type):
    vel = [vX, vY]
    if type == 'linear':
        vel = [vX , vY ]
    if type == 'rotatory':
        vel = [np.cos( 0.1 * frame), np.sin( 0.1 * frame)]
    if type == 'random':
        global theta
        rand = random.randint(0, 360)
        vel = [vX * np.sin(rand), vY * np.cos(rand)]
    return vel



if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        from matplotlib.patches import Rectangle
        import json

        file = open('input5.json')
        input = json.load(file)
        file.close()

        inst = Fluid()

        objects = []
        # We set the static objects in the plot
        for i in input['objects']:
            p = i.split()
            x = int(p[0])
            y = int(p[1])
            v = input['objects'][i]
            objects.append([x, y, v])

        def update_im(frame):
            # We add new density creators in here
            for i in input['density']:
                point = i.split()
                v = input['density'][i]
                if i.find(':') > 0 :
                    point0 = point[0].split(':')
                    point1 = point[1].split(':')
                    x0 = int(point0[0])
                    x1 = int(point0[1])
                    y0 = int(point1[0])
                    y1 = int(point1[1])
                    inst.density[x0:x1, y0:y1] += v
                else:
                    x = int(point[0])
                    y = int(point[1])
                    inst.density[x, y] += v

            # We add velocity vector values in here
            for i in input['velocity']:
                origin = i.split()
                x = int(origin[0])
                y = int(origin[1])
                vector = input['velocity'][i].split()
                xv = int(vector[0])
                yv = int(vector[1])
                aux = velType(frame,xv,yv,input['velocity Type'])
                inst.velo[x, y] = aux

            for o in objects:
                x, y = o[0], o[1]
                w, h = o[2][0], o[2][1]
                #inst.density[y:y+h, x:x+w] = 0
                inst.velo[y:y+h, x:x+w] = [0,0]
                ax.add_patch(Rectangle((x, y), w, h, color=input['obj color']))

            inst.step()
            im.set_array(inst.density)
            q.set_UVC(inst.velo[:, :, 1], inst.velo[:, :, 0])
            # print(f"Density sum: {inst.density.sum()}")
            im.autoscale()

        # Style up the Plot
        
        fig, ax = plt.subplots()
        ax.set_title("Fluid Simulation")

        

        #fig = plt.figure()

        # plot density
        im = plt.imshow(inst.density, vmax=100, interpolation='bilinear', cmap=input['Color Scheme'])

        # plot vector field
        q = plt.quiver(inst.velo[:, :, 1], inst.velo[:, :, 0], scale=10, angles='xy')
        anim = animation.FuncAnimation(fig, update_im,interval=0)
        #anim.save("Simul1.mp4", fps=30)
        plt.show()

    except ImportError:
        import imageio

        frames = 30

        flu = Fluid()

        video = np.full((frames, flu.size, flu.size), 0, dtype=float)

        for step in range(0, frames):
            flu.density[4:7, 4:7] += 100  # add density into a 3*3 square
            flu.velo[5, 5] += [1, 2]

            flu.step()
            video[step] = flu.density

        imageio.mimsave('./video.gif', video.astype('uint8'))
