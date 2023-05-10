import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gfootball.env as football_env
import numpy as np
import tensorflow as tf
import tf.keras.backend as K
from tf.keras.layers import Input, Dense, Flatten
from tf.keras.models import Model
from tf.keras.optimizers import Adam
from tf.keras.applications.mobilenet_v2 import MobileNetV2


def get_model_actor_image(input_dims):
    state_input = Input(shape=input_dims)

    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)

    for layer in feature_extractor.layers:
        layer.trainable = False

    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model


def get_model_critic_image(input_dims):
    state_input = Input(shape=input_dims)

    feature_extractor = MobileNetV2(weights='imagenet', include_top=False)

    for layer in feature_extractor.layers:
        layer.trainable = False

    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='tanh', name='predictions')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model


env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)

state = env.reset()

state_dims = env.observation_space.shape
print(state_dims)

n_actions = env.action_space.n
print(n_actions)

ppo_steps = 128

states = []
actions = []
values = []
masks = []
rewards = []
actions_probs = []
actions_onehot = []

model_actor = get_model_actor_image(input_dims=state_dims)
model_critic = get_model_critic_image(input_dims=state_dims)

for itr in range(ppo_steps):
    state_input = K.expand_dims(state, 0)
    action_dist = model_actor.predict([state_input], steps=1)
    q_value = model_critic.predict([state_input], steps=1)
    action = np.random.choice(n_actions, p=action_dist[0, :])
    action_onehot = np.zeros(n_actions)
    action_onehot[action] = 1

    observation, reward, done, info = env.step(action)
    mask = not done

    states.append(state)
    actions.append(action)
    actions_onehot.append(action_onehot)
    values.append(q_value)
    masks.append(mask)
    rewards.append(reward)
    actions_probs.append(action_dist)

    state = observation

    if done:
        env.reset()

env.close()
