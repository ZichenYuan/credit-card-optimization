# Credit Card Optimization using Multiple Gradient-Based Methods

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load credit card data and user profile
with open("credit_cards.json") as f:
    credit_cards = pd.read_json(f)

with open("user_profiles.json") as f:
    profiles = json.load(f)
    user_profile = profiles["profiles"][1]  # Index 1 corresponds to profile2

# Normalize user profile into a vector of spendings
spending_categories = list(user_profile.keys())
user_spend = np.array([user_profile[cat] for cat in spending_categories])

def compute_cash_reward(card, weight):
    cashback = card['cashback']
    total = 0
    for i, cat in enumerate(spending_categories):
        rate = cashback.get(cat, cashback.get('all', 0))
        total += rate * user_spend[i] * weight
    return total

def compute_point_reward(card, weight):
    if 'point_back' not in card or pd.isna(card['point_back']):
        return 0
    point_back = card['point_back']
    total = 0
    for i, cat in enumerate(spending_categories):
        rate = point_back.get(cat, point_back.get('all', 0))
        total += rate * user_spend[i] * weight
    return total

def compute_cost(card, weight):
    return card['annual_fee'] * weight

# Compute total intro bonus reward for a credit card based on its weight
def compute_intro_bonus(card, weight):
    # Return the weighted intro bonus amount if it exists in card data
    return card.get('intro_bonus', 0) * weight


# loss function to be minimized
# cash_reward_ratio: ratio of cashback rewards to point rewards
# point_reward_ratio: ratio of point rewards to cashback rewards
# intro_bonus: whether to include intro bonus in the loss function  
def loss_function(weights, cash_reward_ratio=0.5, point_reward_ratio=0.5, intro_bonus = True):
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    
    total_cash_reward = 0
    total_point_reward = 0
    total_cost = 0
    total_intro_bonus = 0
    for i, card in credit_cards.iterrows():
        # print(card)
        if 'cashback' in card:
            total_cash_reward += compute_cash_reward(card, weights[i])
        if 'point_back' in card:
            total_point_reward += compute_point_reward(card, weights[i])
        total_cost += compute_cost(card, weights[i])
        if intro_bonus:
            total_intro_bonus += compute_intro_bonus(card, weights[i])
    
    # Normalize ratios to sum to 1
    ratio_sum = cash_reward_ratio + point_reward_ratio
    cash_reward_ratio /= ratio_sum
    point_reward_ratio /= ratio_sum
    
    # Combine rewards with their respective ratios
    total_reward = (cash_reward_ratio * total_cash_reward + 
                   point_reward_ratio * total_point_reward)
    
    return -total_reward + total_cost - total_intro_bonus


def compute_gradient(loss_fn, weights, eps=1e-5):
    grad = np.zeros_like(weights)
    for i in range(len(weights)):
        w_plus = weights.copy()
        w_minus = weights.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        grad[i] = (loss_fn(w_plus) - loss_fn(w_minus)) / (2 * eps)
    return grad

def gradient_descent(lr=0.01, epochs=500):
    weights = np.ones(len(credit_cards)) / len(credit_cards)
    loss_history = []
    for _ in range(epochs):
        grad = compute_gradient(loss_function, weights)
        weights -= lr * grad
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)
        loss_history.append(loss_function(weights))
    return weights, loss_history

def nesterov_accelerated_gradient(lr=0.01, beta=0.9, epochs=500):
    weights = np.ones(len(credit_cards)) / len(credit_cards)
    v = np.zeros_like(weights)
    loss_history = []
    for _ in range(epochs):
        lookahead = weights - beta * v
        grad = compute_gradient(loss_function, lookahead)
        v = beta * v + lr * grad
        weights -= v
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)
        loss_history.append(loss_function(weights))
    return weights, loss_history

def backtracking_line_search(weights, grad, alpha=1.0, beta=0.8, c=1e-4):
    loss = loss_function(weights)
    while loss_function(weights - alpha * grad) > loss - c * alpha * np.dot(grad, grad):
        alpha *= beta
    return alpha

def gradient_descent_backtracking(epochs=500):
    weights = np.ones(len(credit_cards)) / len(credit_cards)
    loss_history = []
    for _ in range(epochs):
        grad = compute_gradient(loss_function, weights)
        lr = backtracking_line_search(weights, grad)
        weights -= lr * grad
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)
        loss_history.append(loss_function(weights))
    return weights, loss_history

def compute_hessian(loss_fn, weights, eps=1e-4):
    n = len(weights)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w_ijp = weights.copy()
            w_ijm = weights.copy()
            w_ipj = weights.copy()
            w_imj = weights.copy()
            w_ijp[i] += eps; w_ijp[j] += eps
            w_ijm[i] += eps; w_ijm[j] -= eps
            w_ipj[i] -= eps; w_ipj[j] += eps
            w_imj[i] -= eps; w_imj[j] -= eps
            hessian[i, j] = (loss_fn(w_ijp) - loss_fn(w_ijm) - loss_fn(w_ipj) + loss_fn(w_imj)) / (4 * eps ** 2)
    return hessian

def newtons_method(epochs=50, damping=True, beta=0.8):
    weights = np.ones(len(credit_cards)) / len(credit_cards)
    loss_history = []
    for _ in range(epochs):
        grad = compute_gradient(loss_function, weights)
        hessian = compute_hessian(loss_function, weights)
        try:
            delta = np.linalg.solve(hessian, grad)
            if damping:
                # Line search for damping factor
                alpha = 1.0
                current_loss = loss_function(weights)
                while loss_function(weights - alpha * delta) > current_loss - 0.5 * alpha * np.dot(grad, delta):
                    alpha *= beta
                delta *= alpha
        except np.linalg.LinAlgError:
            delta = grad  # Fallback to gradient descent step if Hessian is not invertible
        weights -= delta
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)
        loss_history.append(loss_function(weights))
    return weights, loss_history

# Run all optimizers
weights_gd, loss_gd = gradient_descent()
weights_nag, loss_nag = nesterov_accelerated_gradient()
weights_bt, loss_bt = gradient_descent_backtracking()
weights_newton, loss_newton = newtons_method(damping=False)  # Without damping
weights_newton_damped, loss_newton_damped = newtons_method(damping=True)  # With damping

# Plot convergence
plt.figure(figsize=(12, 6))
plt.plot(loss_gd, label="Gradient Descent")
plt.plot(loss_nag, label="Nesterov")
plt.plot(loss_bt, label="Backtracking")
plt.plot(loss_newton, label="Newton's Method (No Damping)")
plt.plot(loss_newton_damped, label="Newton's Method (With Damping)")
plt.title("Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Print optimal weights
methods = ["Gradient Descent", "Nesterov", "Backtracking", "Newton (No Damping)", "Newton (With Damping)"]
results = [weights_gd, weights_nag, weights_bt, weights_newton, weights_newton_damped]

for method, weights in zip(methods, results):
    print(f"\nOptimal Spend Allocation ({method}):")
    for i, card in credit_cards.iterrows():
        print(f"  {card['name']}: {weights[i]:.2f}")

# Print final loss
print("\nFinal Loss Comparison:")
for method, loss in zip(methods, [loss_gd, loss_nag, loss_bt, loss_newton, loss_newton_damped]):
    print(f"  {method}: {loss[-1]:.2f}")
