import logging
import math
import asyncio
import uvloop
import signal
import functools
import numpy as np
from copy import deepcopy
from random import random, randint, choice
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MatpRectangle
import matplotlib.cbook
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from rectpack.geometry import Rectangle as RectpackRectangle
from config import Config

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

log = logging.getLogger('GeneticPacker')
log.setLevel(logging.DEBUG)

def shutdown(loop, executor, signame):
    log.debug('')
    log.debug('Shutdown {0}'.format(signame))
    try:
        tasks = []
        for task in asyncio.Task.all_tasks():
            if task is not asyncio.tasks.Task.current_task():
                tasks.append(task)
                task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    except Exception as e:
        log.debug('Shutdown task error: %s' % e)

class BasePlot(object):
    def __init__(self, plt=None):
        self._plt = plt

    def plot(self):
        if self._plt:
            self._plt.plot([3, 4, 10])

class Line(BasePlot):
    def __init__(self, x, plt=None):
        self.x = x
        BasePlot.__init__(self, plt=plt)

    def plot(self):
        if self._plt:
            self._plt.plot([3, 4, self.x])

class RectangleAlg(object):
    def __init__(self, x, y, width, height, rid=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rid = rid

    def seed(self, kx, ky):
        self.x = kx * np.random.rand((1))[0]
        self.y = ky * np.random.rand((1))[0]
        return self

    def __repr__(self):
        return "RA{} ({}, {}, {}, {})".format(self.rid, self.x, self.y, self.width, self.height)

class Rectangle(RectpackRectangle, BasePlot):
    def __init__(self, x, y, width, height, rid=None, axes=None, fig=None):
        assert(height >=0 and width >=0)
        RectpackRectangle.__init__(self, x, y, width, height, rid=rid)
        self.axes = axes
        self.fig = fig
        self.entitie = 'entitie'
        self.rect = MatpRectangle((self.x, self.y), self.width, self.height, alpha=1, facecolor='none')

    def __repr__(self):
        return "RA{} ({}, {}, {}, {})".format(self.rid, self.x, self.y, self.width, self.height)

    def seed(self,kx,ky):
        self.x = kx * np.random.rand((1))[0]
        self.y = ky * np.random.rand((1))[0]
        return self

    def add_patch(self):
        if self.axes:
            self.axes.add_patch(self.rect)

    def moveto(self, x, y):
        self.move(x, y)
        self.rect.set_xy((x, y))

    def sizeto(self, width, height):
        if width:
            self.width = width
            self.rect.set_width(width)
        if height:
            self.height = height
            self.rect.set_height(height)

    def plot(self):
        self.add_patch()

class BaseAlgorithm(object):
    def __init__(self, config, loop=None, queue=None, axes=None, fig=None):
        self.config = config
        self._loop = loop
        self._queue = queue
        self.queue = asyncio.Queue()
        self._fig = fig
        self._axes = axes
        self._target = None
        self._rects = []
        self._entities = []

    def setTarget(self, x, y, width, height, rid=None):
        self._target = RectpackRectangle(x, y, width, height, rid=rid)

    def setEntities(self, entities):
        kx = self.config.workarea.x
        ky = self.config.workarea.y
        for entitie in entities:
            rect = Rectangle(entitie[0]*kx, entitie[1]*ky, entitie[2]*kx, entitie[3]*ky, entitie[4], axes=self._axes, fig=self._fig)
            self._rects.append(rect)
            rect.add_patch()

    async def start(self):
        try:
            for i in range(self.config.size):
                self._entities.append(self.clone(self.seed()))

            for i in range(self.config.iterations):
                self.internalGenState = {}
                pop = []
                for entitie in self._entities:
                    pop.append({
                        'fitness': self.fitness(entitie),
                        'entitie': entitie
                    })

                pop = sorted(pop, key=functools.cmp_to_key(self.optimize))
                stats = {
                    "maximum": pop[0]['fitness'],
                    "minimum": pop[len(pop)-1]['fitness'],
                    "mean": '',
                    "stdev": ''
                }
                isFinished = (i == self.config.iterations-1)
                if (isFinished):
                    break

                newPop = []
                if self.config.fittestAlwaysSurvives:
                    newPop.append(pop[0]['entitie'])
                while len(newPop) < self.config.size:
                    if random() <= self.config.crossover and len(newPop)+1 < self.config.size:
                        parents = self.select2(pop)
                        children = self.crossover(self.clone(parents[0]), self.clone(parents[1]))
                        children = [self.mutateOrNot(x) for x in children]
                        newPop.append(children[0])
                        newPop.append(children[1])
                    else:
                        newPop.append(self.mutateOrNot(self.select1(pop)))

                self._entities = newPop
                await self.queue.put({'updatemodel': True, 'rects': pop[0]['entitie']})
                await asyncio.sleep(0.1)
        except Exception as e:
            log.error('Error: %s' % e)

    async def updategraph(self):
        kx = self.config.workarea.x
        ky = self.config.workarea.y
        while True:
            item = await self.queue.get()
            if item['updatemodel']:
                rects = item['rects']
                for rect in rects:
                    for entitie in self._rects:
                        if entitie.rid == rect.rid:
                            entitie.moveto(rect.x * kx, rect.y * ky)
                await self._queue.put('update')
            await asyncio.sleep(1)

    def mutateOrNot(self, entitie):
        if random() <= self.config.mutation:
            res = self.mutate(self.clone(entitie))
        else:
            res = entitie
        return res

    def clone(self, obj):
        obj = deepcopy(obj)
        return obj

    def centerplacment(self):
        for rect in self._rects:
            x = self.config.workarea.x / 2
            y = self.config.workarea.y /2
            rect.moveto(x, y)

    def randomplacment(self):
        for rect in self._rects:
            x = self.config.workarea.x * np.random.rand((1))[0]
            y = self.config.workarea.y * np.random.rand((1))[0]
            rect.moveto(x, y)

    def fitness(self, entities):
        prev = {'overlap': 0,
                'containarea':0,
                'samplesarea': 0,
                'sum': 0,
                'maxx': 0,
                'maxy': 0
               }

        mainarea = self.config.workarea.x * self.config.workarea.y

        for j in range(len(entities)):
            cur = entities[j]
            for i in range(j + 1, len(entities)):
                prev['overlap'] += self.intersection(entities[i], entities[j])

            prev['samplesarea'] += cur.width * cur.height
            prev['containarea'] += self.containarea(cur)
            prev['sum'] += cur.x + cur.y
            prev['maxx'] = max(cur.x + cur.width, prev['maxx'])
            prev['maxy'] = max(cur.y + cur.height, prev['maxy'])

        prev['samplesarea'] = 1 - (prev['containarea'] / prev['samplesarea'])
        prev['containarea'] = 1 - (prev['containarea'] / mainarea)

        # return [prev['overlap'], prev['samplesarea'], prev['containarea'] ]
        return [prev['overlap'], prev['maxx'] * prev['maxy'], prev['sum']]

    def containarea(self, rect):
        mainx = self.config.workarea.x
        mainy = self.config.workarea.y

        if (rect.y >= 0 and rect.x <= 0 and  rect.y+rect.height <= mainy and rect.x+rect.width  <= mainx):
            containarea = rect.height * rect.width
        else:
            containarea = 0

        return containarea

    def seed(self):
        kx = self.config.workarea.x
        ky = self.config.workarea.y
        return [RectangleAlg(random(), random(), x.width/kx, x.height/ky, rid=x.rid) for x in self._rects]

    def mutate(self, entitys):
        def change(n, drift):
            n += random() * drift
            return 0 if n < 0 else n

        def copychange(r, d):
            dir = random() > .5
            x  = change(r.x, d) if dir else r.x
            y  = change(r.y, d) if dir else r.y
            return RectangleAlg(x,y, r.width, r.height, rid=r.rid)

        drift = ((random() - 0.5) * 2) * self.config.drift
        copy =  [copychange(x, drift) for x in entitys]
        while random() > .5:
            i = randint(0, len(entitys)-1)
            j = randint(0, len(entitys)-1)
            copy[i].x = entitys[j].x
            copy[i].y = entitys[j].y
            copy[j].x = entitys[i].x
            copy[j].y = entitys[i].y

        return copy

    def intersection(self, r1, r2):
        if r1.x < r2.x:
            x = r2.x
            dx = r1.x + r1.width - r2.x
        else:
            x = r1.x
            dx = r2.x + r2.width - r1.x
        if dx < 0:
            dx = 0
        if dx > r1.width:
            dx = r1.width
        if dx > r2.width:
            dx = r2.width

        if r1.y < r2.y:
            y = r2.y
            dy = r1.y + r1.height - r2.y
        else:
            y = r1.y
            dy = r2.y + r2.height - r1.y
        if dy < 0:
            dy = 0
        if dy > r1.height:
            dy = r1.height
        if dy > r2.height:
            dy = r2.height

        return dx * dy

    def crossover(self, mother, father):
        end = randint(0, len(mother))
        son = deepcopy(father)
        daughter = deepcopy(mother)

        for i in range(end):
            son[i] = mother[i]
            daughter[i] = father[i]
        return [son, daughter]

    def optimize(self, a, b):
        a = a['fitness']
        b = b['fitness']
        sort = 1
        for i in range(len(a)):
            if a[i] < b[i]:
                return -1
            elif a[i] > b[i]:
                return 1
        return -1

    def select1(self, pop):
        index = randint(0, len(pop)-1)
        return pop[index]['entitie']

    def select2(self, pop):
        entity1 = pop[0]['entitie']
        entity2 = self.select1(pop)
        return [entity1, entity2]

class DataAnalysis():
    _fig = None
    _axes = None

    def __init__(self):
        self.config = Config().cfgdata
        self._loop = uvloop.new_event_loop()
        asyncio.set_event_loop(self._loop)
        executor = ProcessPoolExecutor(max_workers=4) # executor = ThreadPoolExecutor(max_workers=15)
        self._loop.set_default_executor(executor)
        self._queue = asyncio.Queue()

        self.initgraph()
        self._algorithm = self.initalgorithm()

        for signame in ('SIGINT', 'SIGTERM'):
            self._loop.add_signal_handler(getattr(signal, signame),
                                    functools.partial(shutdown, self._loop, executor, signame))

    def initgraph(self):
        bin_target = (self.config.workarea.x, self.config.workarea.y)
        bin_pull = (self.config.workarea.x * 2, self.config.workarea.x * 2)

        self._fig = plt.figure(211)
        plt.title('Genetic Programming Dimensional Bin Packing')
        self._axes = plt.axes(xlim=(-10, self.config.workarea.x * 2 + 60), ylim=(-10, self.config.workarea.y * 2 + 10))

        self._axes.add_patch(MatpRectangle((0, 0), bin_target[0], bin_target[1], alpha=1, facecolor='none', edgecolor="b"))
        self._axes.annotate('target', (0, -2), color='b', weight='bold', fontsize=8, ha='left', va='center')

        self._axes.add_patch(MatpRectangle((55, 0), bin_pull[0], bin_pull[1], alpha=1, facecolor='none', edgecolor="r"))
        self._axes.annotate('pull', (55, -2), color='r', weight='bold', fontsize=8, ha='left', va='center')

    def initalgorithm(self):
        bin_target = (self.config.workarea.x, self.config.workarea.y)
        bin_pull = (self.config.workarea.x * 2, self.config.workarea.x * 2)
        algorithm = BaseAlgorithm(self.config, loop=self._loop, queue=self._queue,
                                  axes=self._axes, fig=self._fig)
        algorithm.setTarget(0, 0, bin_target[0], bin_target[1], rid=0)

        qsample = randint(self.config.qsamplemin, self.config.qsamplemax)
        rectangles = np.random.rand(qsample, 4)
        rectangles = [(x, y, w/2, h/2, n) for (n, (x, y, w, h)) in enumerate(rectangles)]

        algorithm.setEntities(rectangles)
        return algorithm

    def start(self):
        try:
            asyncio_tasks = asyncio.gather(self.plot_reward(), self._algorithm.start(), self._algorithm.updategraph())
            self._loop.run_until_complete(asyncio_tasks)
        except KeyboardInterrupt:
            log.debug("Attempting graceful shutdown, press Ctrl+C again to exit…")
        except asyncio.CancelledError:
            log.debug("Task exit by Ctrl+C …")
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def plot_reward(self):
        plt.pause(0.01)
        while True:
            item = await self._queue.get()
            if isinstance(item, BasePlot):
                print(item)
                item.plot()
                plt.pause(0.01)
            elif item == 'update':
                plt.pause(0.01)

if __name__ == '__main__':
    da = DataAnalysis()
    da.start()