#!//usr/bin/env python3

import numpy as np

def HMC(U, grad_U, epsilon, L , current_q):
    q = current_q.copy()
    p = np.random.normal(0, 1, size = q.shape[0])
    current_p = p.copy()

    p = p - epsilon * grad_U(q) / 2

    for i in range(L):
        q = q + epsilon * p

        if i != L:
            p = p - epsilon * grad_U(q)
        
        #print("In L",U(q) + np.sum(p**2) / 2)

    p = p - epsilon * grad_U(q) / 2

    p = -p

    current_U = U(current_q)
    current_K = np.sum(current_p**2) / 2

    proposed_U = U(q)
    proposed_K = np.sum(p**2) / 2
    
    print(current_U, current_K, proposed_U, proposed_K)
    print(current_U - proposed_U + current_K - proposed_K)

    if np.random.uniform(0,1) < np.exp(current_U - proposed_U + current_K - proposed_K):
        return q
    else:
        print("r")
        return current_q
