"""
Worked examples from the lectures, checked with code (exam practice).
Run:  python lecture_worked_examples.py
"""
import math

print("=" * 60)
print("1) Bayesian decoding (Lecture: Encoding & Decoding)")
# S1 = vertical grating, S2 = horizontal grating, observed R = 8 spikes
p_s1, p_s2 = 0.5, 0.5
p_r_s1, p_r_s2 = 0.30, 0.10
p_r = p_r_s1 * p_s1 + p_r_s2 * p_s2
print(f"P(R=8) = 0.30*0.5 + 0.10*0.5 = {p_r:.2f}")
print(f"P(S1|R=8) = 0.15 / {p_r:.2f} = {p_r_s1 * p_s1 / p_r:.2f}")
print(f"P(S2|R=8) = 0.05 / {p_r:.2f} = {p_r_s2 * p_s2 / p_r:.2f}  -> decoder chooses S1 (vertical)")

print("=" * 60)
print("2) Forward pass of one neuron (Lecture: Computing with Networks)")
x = [0.5, -0.6, 0.2]
w = [0.8, -0.5, 0.3]
b = 0.1
z = sum(xi * wi for xi, wi in zip(x, w)) + b
print(f"z = 0.5*0.8 + (-0.6)*(-0.5) + 0.2*0.3 + 0.1 = {z:.2f}")
print(f"step (z >= 0)      -> {1 if z >= 0 else 0}")
print(f"sigmoid 1/(1+e^-z) -> {1 / (1 + math.exp(-z)):.3f}")
for theta in (0.5, 1.0):
    print(f"spiking, threshold theta = {theta} -> {'spike (1)' if z >= theta else 'no spike (0)'}")

print("=" * 60)
print("3) Entropy (Lecture: Entropy & Mutual Information)")
H = lambda p: abs(-sum(q * math.log2(q) for q in p if q > 0))
for p in ([0.5, 0.5], [1.0, 0.0], [0.5, 0.25, 0.25], [0.95, 0.05], [0.25] * 4, [0.7, 0.1, 0.1, 0.1]):
    print(f"H{p} = {H(p):.3f} bits")

print("=" * 60)
print("4) Poisson input rate in Diehl & Cook (Lecture 8)")
for pix in (0, 64, 128, 255):
    r = pix / 4
    print(f"pixel {pix:>3} -> rate {r:6.2f} Hz -> expected spikes in 350 ms = {r * 0.35:.1f}")
print("P(n spikes in T) = (rT)^n e^(-rT) / n!   e.g. r = 20 Hz, T = 0.1 s:")
for n in range(5):
    print(f"   P({n}) = {(2 ** n) * math.exp(-2) / math.factorial(n):.3f}")

print("=" * 60)
print("5) Backpropagation, one step (Lecture: Computing with Networks, 2-2-2 network)")
s = lambda v: 1 / (1 + math.exp(-v))
i1, i2 = 0.05, 0.10
w1, w2, w3, w4, w5, w6, w7, w8 = 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
t1, t2 = 0.01, 0.99
h1 = s(w1 * i1 + w2 * i2 + b1)
h2 = s(w3 * i1 + w4 * i2 + b1)
o1 = s(w5 * h1 + w6 * h2 + b2)
o2 = s(w7 * h1 + w8 * h2 + b2)
E = 0.5 * (t1 - o1) ** 2 + 0.5 * (t2 - o2) ** 2
print(f"forward: h1 = {h1:.4f}, h2 = {h2:.4f}, o1 = {o1:.4f}, o2 = {o2:.4f}, E_total = {E:.4f}")
dE_do1 = o1 - t1
do1_dnet = o1 * (1 - o1)
dnet_dw5 = h1
grad = dE_do1 * do1_dnet * dnet_dw5
print(f"chain rule: dE/dw5 = (o1 - t1) * o1(1 - o1) * h1 = {dE_do1:.4f} * {do1_dnet:.4f} * {dnet_dw5:.4f} = {grad:.4f}")
print(f"update (learning rate 0.5): w5 = 0.40 - 0.5 * {grad:.4f} = {w5 - 0.5 * grad:.4f}")
