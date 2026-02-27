import time

import arcade
import jax
import jax.numpy as jnp
from arcade.shape_list import ShapeElementList
from jax import lax, random


class State:
    ATTRACT_CONSTANT = 50.0
    ATTRACT_POWER = 2
    REPULSION_CONSTANT = 7.0 * 10**5
    REPULSION_POWER = 4

    def __init__(self, num_atoms):
        key = random.key(2)
        keys = random.split(key, num_atoms)
        positions = jax.vmap(
            lambda k: random.uniform(k, (2,), minval=-400.0, maxval=400.0)
        )(keys)
        self.xs = positions[:, 0]
        self.ys = positions[:, 1]

    @staticmethod
    @jax.jit
    def gravity(x, y, xs, ys):
        attraction = jax.vmap(
            lambda x2, y2: (
                (
                    jnp.array([x2 - x, y2 - y])
                    / (
                        jnp.sqrt((x2 - x) ** 2 + (y2 - y) ** 2) ** State.ATTRACT_POWER
                        + 1e-6
                    )
                )
                * State.ATTRACT_CONSTANT
            )
        )(xs, ys).sum(axis=0)
        return attraction

    @staticmethod
    @jax.jit
    def between(x1, y1, x2, y2):
        delta = jnp.array([x1, y1]) - jnp.array([x2, y2])
        distance = jnp.sqrt(delta[0] ** 2 + delta[1] ** 2)
        return lax.cond(
            distance > 1.0,
            lambda: (
                State.REPULSION_CONSTANT * delta / (distance**State.REPULSION_POWER)
            ),
            lambda: jnp.array([0.0, 0.0]),
        )

    @staticmethod
    @jax.jit
    def repulsion(x, y, xs, ys):
        return jax.vmap(State.between, in_axes=(None, None, 0, 0))(x, y, xs, ys).sum(
            axis=0
        )

    @staticmethod
    @jax.jit
    def update_step(xs, ys, dt=1.0):
        deltas = jax.vmap(
            lambda x, y, xs, ys: (
                (State.gravity(x, y, xs, ys) + State.repulsion(x, y, xs, ys)) * dt
            ),
            in_axes=(0, 0, None, None),
        )(xs, ys, xs, ys)
        return xs + deltas[:, 0], ys + deltas[:, 1]

    MICROSTEPS = 16

    def update(self, dt=1.0):
        for _ in range(State.MICROSTEPS):
            self.xs, self.ys = self.update_step(self.xs, self.ys, dt / State.MICROSTEPS)


def main():
    state = State(num_atoms=6_00)
    window = arcade.Window(1600, 1600, "Particle Simulation", resizable=True)
    camera = arcade.Camera2D()

    pan_speed = 400.0
    zoom_speed = 1.1
    min_zoom = 0.1
    max_zoom = 10.0
    pressed_keys = set()

    fps = 0.0

    @window.event
    def on_draw():
        begin = time.perf_counter()

        window.clear(color=arcade.color.BLACK)
        camera.use()

        after = time.perf_counter()
        delta_t = after - begin
        print(f"Draw took {delta_t * 1000:.4f} ms")

    @window.event
    def on_key_press(symbol, modifiers):
        pressed_keys.add(symbol)

    @window.event
    def on_key_release(symbol, modifiers):
        pressed_keys.discard(symbol)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        if scroll_y < 0:
            camera.zoom = min(camera.zoom * zoom_speed, max_zoom)
        elif scroll_y > 0:
            camera.zoom = max(camera.zoom / zoom_speed, min_zoom)

    def on_update(delta_time):
        nonlocal fps
        print(fps)

        begin = time.perf_counter()
        state.update(delta_time)
        after = time.perf_counter()
        delta_t = after - begin
        print(f"Update took {delta_t * 1000:.4f} ms")
        if delta_time > 0:
            fps = 1.0 / delta_time

        move_x = 0.0
        move_y = 0.0

        if arcade.key.W in pressed_keys or arcade.key.UP in pressed_keys:
            move_y += pan_speed * delta_time
        if arcade.key.S in pressed_keys or arcade.key.DOWN in pressed_keys:
            move_y -= pan_speed * delta_time
        if arcade.key.A in pressed_keys or arcade.key.LEFT in pressed_keys:
            move_x -= pan_speed * delta_time
        if arcade.key.D in pressed_keys or arcade.key.RIGHT in pressed_keys:
            move_x += pan_speed * delta_time

        if move_x != 0.0 or move_y != 0.0:
            camera.position = (camera.position[0] + move_x, camera.position[1] + move_y)

    window.on_update = on_update
    window.set_update_rate(1.0 / 60.0)
    arcade.run()


if __name__ == "__main__":
    main()
