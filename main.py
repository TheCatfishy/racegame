import pygame
import os
import sys
import math
import neat

W_WIDTH = 1920
W_HEIGHT = 1080
CHECK_POINTS = 12

pygame.init()
screen = pygame.display.set_mode((W_WIDTH, W_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Comic Sans MS', 30)
ff_font = pygame.font.SysFont('Comic Sans MS', 100)
pygame.display.set_caption("shit raceGimma")
curr_gen = 0
fast_forward = False


class Time:
	def __init__(self, time):
		self.total = time
		self.ms = time % 1000
		self.seconds = int((time / 1000) % 60)
		self.minutes = int((time / 60000) % 60)
		self.hours = int((time / 3600000))

	def __sub__(a, b):
		return Time(a.total - b.total)

	def __str__(self):
		return f"{self.hours}:{self.minutes}:{self.seconds}.{self.ms}"


class Car:
	def __init__(self, genome, config):
		self.original_image = pygame.image.load('racecar.png')
		self.image = self.original_image
		self.rect = self.image.get_rect()
		self.rect.y = 450
		self.rect.x = 1750
		self.lastTime = Time(0)
		self.Time = Time(0)
		self.timeArray = [Time(0)] * CHECK_POINTS
		self.lineArray = [False] * CHECK_POINTS
		self.drag = 0.1
		self.speed_increase = 0.5
		self.maxSpeed = 5
		self.vel = 0
		self.angle = 0
		self.mask = pygame.mask.from_surface(self.image)
		self.front_left = 0
		self.front = 0
		self.front_right = 0
		self.rightDist = 0
		self.leftDist = 0
		self.moved = 0
		self.brain = neat.nn.FeedForwardNetwork.create(genome, config)
		self.genome = genome
		self.genome.fitness = 0
		self.front_left_tt = 0
		self.front_left_ss = 0
		self.front_right_tt = 0
		self.rightDist_ss = 0

	def move(self):
		if self.vel > 0:
			self.vel -= self.drag
		if self.vel < 0:
			self.vel = 0
		self.rect.y += self.vel * math.cos(math.radians(self.angle))
		self.rect.x += self.vel * math.sin(math.radians(self.angle))

	def forward(self):
		self.vel += self.speed_increase
		if self.vel > self.maxSpeed:
			self.vel = self.maxSpeed

	def backward(self):
		self.vel -= self.speed_increase
		if self.vel < 0:
			self.vel = 0

	def left(self):
		self.angle += 2
		self.angle = self.angle % 360
		self.update_image()

	def right(self):
		self.angle -= 2
		self.angle = self.angle % 360
		self.update_image()

	def decide(self):
		output = self.brain.activate((self.vel, self.front_left, self.front, self.front_right, self.rightDist, self.leftDist, self.front_left_tt , self.front_left_ss , self.front_right_tt , self.rightDist_ss))
		if (output[0] > 0.5):
			self.forward()
		if (output[1] > 0.5):
			self.left()
		if (output[2] > 0.5):
			self.right()
		if (output[3] > 0.5):
			self.backward()

	def update_image(self):
		self.image = pygame.transform.rotate(self.original_image, self.angle)
		x, y = self.rect.center
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)
		self.mask = pygame.mask.from_surface(self.image)

	def draw(self):
		screen.blit(self.image, self.rect)
		x, y = self.rect.center
		x_len = math.sin(math.radians(self.angle - 45)) * self.front_left
		y_len = math.cos(math.radians(self.angle - 45)) * self.front_left
		pygame.draw.line(screen, (255, 0, 0), (x, y), (x + x_len, y + y_len), 3)
		x_len = math.sin(math.radians(self.angle)) * self.front
		y_len = math.cos(math.radians(self.angle)) * self.front
		pygame.draw.line(screen, (255, 0, 0), (x, y), (x + x_len, y + y_len), 3)
		x_len = math.sin(math.radians(self.angle + 45)) * self.front_right
		y_len = math.cos(math.radians(self.angle + 45)) * self.front_right
		pygame.draw.line(screen, (255, 0, 0), (x, y), (x + x_len, y + y_len), 3)
		x_len = math.sin(math.radians(self.angle + 90)) * self.rightDist
		y_len = math.cos(math.radians(self.angle + 90)) * self.rightDist
		pygame.draw.line(screen, (255, 0, 0), (x, y), (x + x_len, y + y_len), 3)
		x_len = math.sin(math.radians(self.angle - 90)) * self.leftDist
		y_len = math.cos(math.radians(self.angle - 90)) * self.leftDist
		pygame.draw.line(screen, (255, 0, 0), (x, y), (x + x_len, y + y_len), 3)

	def cast(self, mask, angle):
		x, y = self.rect.center
		ray_len = 0
		x_len = 0
		y_len = 0
		while (not mask.get_at((x + int(x_len), y + int(y_len)))):
			ray_len += 1
			x_len = math.sin(math.radians(angle)) * ray_len
			y_len = math.cos(math.radians(angle)) * ray_len


		return ray_len

	def ray_cast(self, mask):
		self.front_left = self.cast(mask, self.angle - 45)
		self.front = self.cast(mask, self.angle)
		self.front_right = self.cast(mask, self.angle + 45)
		self.leftDist = self.cast(mask, self.angle - 90)
		self.rightDist = self.cast(mask, self.angle + 90)
		self.front_left_tt = self.cast(mask, self.angle - 66)
		self.front_left_ss = self.cast(mask, self.angle-22)
		self.front_right_tt = self.cast(mask, self.angle + 22)
		self.rightDist_ss = self.cast(mask, self.angle + 66)


class Checkpoint:
	def __init__(self, index):
		self.index = index
		self.image = pygame.image.load(f"line{index + 1}.png")
		self.mask = pygame.mask.from_surface(self.image)


class Track:
	def __init__(self):
		self.original_image = pygame.image.load("track4.png").convert_alpha()
		self.finishLine = pygame.image.load("finishLine.png").convert_alpha()
		self.concrete = pygame.image.load("concrete.png").convert_alpha()
		self.image = self.original_image
		self.rect = self.image.get_rect()
		self.rect2 = self.concrete.get_rect()
		self.rect3 = self.finishLine.get_rect()
		self.rect.x = 0
		self.rect.y = 0
		self.lines = []
		for i in range(CHECK_POINTS):
			self.lines.append(Checkpoint(i))

		self.mask = pygame.mask.from_surface(self.image, 50)
		self.maskfl = pygame.mask.from_surface(self.finishLine, 50)

	def draw(self):
		screen.blit(self.image, self.rect)
		screen.blit(self.concrete, self.rect2)
		screen.blit(self.finishLine, self.rect3)


class RaceGame:
	def __init__(self, cars, time):
		self.time = Time(0)
		self.start_time = time
		self.track = Track()
		self.cars = cars


	def cars_decide(self):
		for car in self.cars:
			car.ray_cast(self.track.mask)
			car.decide()

	def handle_events(self):
		global fast_forward
		for event in pygame.event.get():
			if (event.type == pygame.QUIT):
				sys.exit()
			elif (event.type == pygame.KEYDOWN):
				if (event.key == pygame.K_SPACE):
					fast_forward = not fast_forward
					if (fast_forward == True):
						self.draw_ffscreen()
				elif (event.key == pygame.K_k):
					already_destroyed = False
					for blyat in self.cars:
						if not already_destroyed:
							self.cars.remove(blyat)
							already_destroyed = True

	def draw_ffscreen(self):
		screen.fill((255, 255, 255))
		ff_text = ff_font.render(">>> Fast Forwarding >>>", False, (0, 0, 0))
		screen.blit(ff_text, (200, 350))
		pygame.display.update()


	def move_entities(self):
		for car in self.cars:
			car.move()

	def check_finish(self):
		for car in self.cars:
			bx, by = (car.rect[0], car.rect[1])
			if not all(car.lineArray):
				return
			if not self.track.maskfl.overlap_area(car.mask, (bx, by)):
				return
			car.genome.fitness += 10
			car.lineArray = [False] * CHECK_POINTS

	def check_collision(self):
		for car in self.cars:
			if (self.track.mask.overlap_area(car.mask, (car.rect[0], car.rect[1]))):
				self.cars.remove(car)
		return len(self.cars) == 0

	def check_lines(self):
		for line in self.track.lines:
			self.check_line(line)

	def check_line(self, line):
		for car in self.cars:
			bx, by = (car.rect[0], car.rect[1])
			if line.mask.overlap_area(car.mask, (bx, by)):
				if (car.lineArray[line.index] == False):
					car.lineArray[line.index] = True
					car.genome.fitness += 1

	def draw_entities(self):
		screen.fill((255, 255, 255))
		self.track.draw()
		for car in self.cars:
			car.draw()
		time = font.render(f"time:      {self.time}", False, (255, 255, 255))
		screen.blit(time, (1600, 40))
		olist = self.track.maskfl.outline()
		pygame.draw.lines(screen, (200, 150, 150), True, olist)
		olist = self.track.maskfl.outline()
		pygame.draw.lines(screen, (200, 150, 150), True, olist)
		pygame.display.update()

	def update_time(self, time):
		self.time = Time(time)

	def destroy_afkers(self):
		curr_time = self.time.total - self.start_time.total
		for car in self.cars:
			if math.floor(curr_time / 4000) > car.genome.fitness and car.genome.fitness < 22:
				self.cars.remove(car)


def train(genomes, config):
	global curr_gen, fast_forward
	curr_score = 0
	cars = []
	for genome_id, genome in genomes:
		cars.append(Car(genome, config))

	game = RaceGame(cars, Time(pygame.time.get_ticks()))
	while True:
		curr_gen += 1
		game.handle_events()
		game.cars_decide()
		game.move_entities()
		game.destroy_afkers()
		if (game.check_collision()):
			break
		if (not fast_forward):
			game.draw_entities()
		clock.tick(120)
		game.update_time(pygame.time.get_ticks())
		game.check_lines()
		game.check_finish()


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config1.txt')
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
							neat.DefaultStagnation, config_path)
p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())

winner = p.run(train, 500)

# show final stats
print('\nBest genome:\n{!s}'.format(winner))
