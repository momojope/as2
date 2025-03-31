import torch
from torch import Tensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
from typing import List, Dict, Union

BOS_TOKEN = "BOS"
EOS_TOKEN = "EOS"
EQ_TOKEN = "="
PAD_TOKEN = "PAD"

def modular_division(a: int, b: int, p: int, q: int):
    """
    #https://www.geeksforgeeks.org/modular-division/

    Compute  (a / b) % q for a, b in Z/pZ.

    Fermat's little theorem: If p is prime and does not divide n, then n^{p−1} ≡ 1 (mod p).

    A simple consequence of Fermat's little theorem is that if n is prime, then n^{−1} ≡ n^{p−2} (mod p)
    is the multiplicative inverse of 0 < n < p.
    More generally, from Euler's theorem, if n and m are coprime, then n^{−1} ≡ c^{φ(m)}−1 (mod m).
    """
    if math.gcd(b, p) != 1: return -1
    a = a % p
    # If b and p are relatively prime, then modulo inverse is b^(p-2) mod p
    b_inverse = pow(b, p - 2, p)
    return (a*b_inverse) % q

########################################################################################
########################################################################################

class Tokenizer:
    """Stores the list of token text to token id mappings and converts between them"""
    def __init__(self, tokens: List[str]) -> None:
        self.itos = tokens # list of tokens
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)} # tokens to ids

    def encode(self, obj: Union[str, List[str]]) -> List[Tensor]:
        """
        Encodes a string or list of strings into token indices.
        
        :param obj: the string or list of strings to convert
        :returns: a tensor of the token ids
        """
        if isinstance(obj, str):
            return LongTensor([self.stoi[t] for t in obj.split(" ")])
        elif isinstance(obj, list):
            #return torch.stack([LongTensor([self.stoi[t] for t in s.split(" ")]) for s in obj], dim=0)
            return [LongTensor([self.stoi[t] for t in s.split(" ")]) for s in obj]
        else:
            raise NotImplementedError

    def decode(self, tensor: Tensor) -> str:
        """
        Convert a tensor of token ids into a string of text

        :param tensor: a tensor of the token ids
        :returns: string of these tokens.
        """
        return " ".join([self.itos[i] for i in tensor.long()])

    def __len__(self) -> int:
        return len(self.itos)

########################################################################################
########################################################################################

def get_arithmetic_dataset(
    p: int,
    q: int,
    operator: str,
    r_train: float,
    operation_orders: Union[int, List[int]] = 2,
    is_symmetric: bool = False,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Generates a dataset of arithmetic expressions with variable operand counts:
    - Binary operations: "a op b = r" with r = (a op b) % q for a, b in Z/pZ.
    - Ternary operations: "a op b op c = r" with r = (a op b op c) % q for a, b, c in Z/pZ.

    Args:
        p (int): The modulo for arithmetic operations.
        q (int): The modulo for the results
        operator (str): The operation to use ("+", "-", "*", "/").
            If operator is "/", then q must be a prime number and the dataset will only contain equations where b is relatively prime to p.
        r_train (float): Train-validation split ratio.
        operation_orders (Union[int, List[int]]): Operation complexity to include.
            - 2: Binary operation (a op b)
            - 3: Ternary operation (a op b op c)
            - [2, 3]: Both types
        is_symmetric (bool): If True, ensures a ≤ b for binary operations and/or a ≤ b ≤ c for ternary operations.
        shuffle (bool): If True, shuffles the dataset.
        seed (int): Random seed for shuffling.

    Returns:
        Tuple[Tuple[TensorDataset, TensorDataset], Tokenizer, int, int]: 
        (train_dataset, valid_dataset), tokenizer, max_length, padding_index
    """
    assert p > 0 and q > 0, "p and q must be positive integers."
    assert operator in ["+", "-", "*", "/"], "Invalid operator. Must be one of '+', '-', '*', '/'."
    assert 0 < r_train <= 1.0, "r_train must be in the range (0, 1]."

    # Ensure operation_orders is a list
    if isinstance(operation_orders, int):
        operation_orders = [operation_orders]

    assert all(o in [2, 3] for o in operation_orders), "operation_orders must be 2, 3 or [2, 3]."

    tokens = [str(i) for i in range(max(p, q))] + [operator, EQ_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
    tokens = list(dict.fromkeys(tokens))  # Remove duplicates
    tokenizer = Tokenizer(tokens=tokens)

    min_bc = 1 if operator == "/" else 0
    equations = []    

    if 2 in operation_orders:
        pairs = [(a, b) for a in range(p) for b in range(max(a, min_bc) if is_symmetric else min_bc, p)]
        if operator in ["+", "-", "*"]:
            equations += [
                f"{a} {operator} {b} {EQ_TOKEN} {(eval(f'({a} {operator} {b}) % {q}'))}"
                for a, b in pairs
            ]
        elif operator == "/":
            # equations += [
            #     f"{a} {operator} {b} {EQ_TOKEN} {modular_division(a, b, p, q)}"
            #     for a, b in pairs
            # ]
            for a, b in pairs:
                #if b == 0: continue
                c = a
                a = (b * c) % q
                equations.append(f"{a} {operator} {b} {EQ_TOKEN} {c}")

    if 3 in operation_orders:
        triplets = [
            (a, b, c) for a in range(p) 
            for b in range(max(a, min_bc) if is_symmetric else min_bc, p) 
            for c in range(max(b, min_bc) if is_symmetric else min_bc, p)
        ]
        if operator in ["+", "-", "*"]:
            equations += [
                f"{a} {operator} {b} {operator} {c} {EQ_TOKEN} {(eval(f'({a} {operator} {b} {operator} {c}) % {q}'))}"
                for a, b, c in triplets
            ]
        elif operator == "/":
            # equations += [
            #     f"{a} {operator} {b} {operator} {c} {EQ_TOKEN} {modular_division(modular_division(a, b, p, q), c, p, q) }"
            #     for a, b, c in triplets
            # ]
            for a, b, c in triplets:
                #if b == 0 or c == 0: continue
                d = a
                a = (b * c * d) % q
                equations.append(f"{a} {operator} {b} {operator} {c} {EQ_TOKEN} {d}")

    if shuffle:
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(equations)

    #equations = [BOS_TOKEN + " " + eq for eq in equations]
    equations = [BOS_TOKEN + " " + eq + " " + EOS_TOKEN for eq in equations]

    encoded_data = tokenizer.encode(equations)

    # Find the maximum sequence length
    max_length = max(len(seq) for seq in encoded_data)

    if operation_orders == [2, 3] :
        # Pad sequences
        padded_data = torch.full((len(encoded_data), max_length), tokenizer.stoi[PAD_TOKEN], dtype=torch.long)
        mask = torch.zeros((len(encoded_data), max_length), dtype=torch.long)
        for i, seq in enumerate(encoded_data):
            length = len(seq)
            padded_data[i, :length] = seq
            mask[i, :length] = 1  # Mask valid tokens
    else:
        # No need for padding since every sequence has the same length
        padded_data = torch.stack(encoded_data, dim=0) # (N, max_length)
        mask = torch.ones_like(padded_data) # (N, max_length)
    
    # Find "=" positions in the target 
    eq_token_index = tokenizer.stoi[EQ_TOKEN]
    eq_positions = torch.where(padded_data == eq_token_index)[1] - 1
    max_length = max_length - 1
    padding_index = tokenizer.stoi[PAD_TOKEN]

    # print(encoded_data)
    # print(padded_data)
    # print(mask)
    # print(eq_positions)

    # Train-validation split
    N_total = len(equations)
    N_train = round(N_total * r_train)
    train_data, valid_data = padded_data[:N_train], padded_data[-1 if N_train == N_total else N_train:]
    train_mask, valid_mask = mask[:N_train], mask[-1 if N_train == N_total else N_train:]
    train_eq_pos, valid_eq_pos = eq_positions[:N_train], eq_positions[-1 if N_train == N_total else N_train:]

    # inputs, labels, eq_positions, masks
    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:], train_eq_pos, train_mask[:, :-1])
    valid_dataset = TensorDataset(valid_data[:, :-1], valid_data[:, 1:], valid_eq_pos, valid_mask[:, :-1])

    return (train_dataset, valid_dataset), tokenizer, max_length, padding_index

########################################################################################
########################################################################################

if __name__ == "__main__":
    p = 2
    operator = "+"
    r_train = 1.
    seed = 42
    operation_orders = [2]  # Include both binary and ternary operations

    (train_dataset, valid_dataset), tokenizer, max_length, padding_index = get_arithmetic_dataset(
        p, p, operator, r_train, operation_orders, is_symmetric=False, shuffle=False, seed=seed
    )

    batch_size = 20
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(tokenizer.stoi)
    for batch in train_dataloader:
        inputs, targets, eq_positions, mask = batch
        eq_token_indexes = targets[torch.arange(targets.shape[0]), eq_positions]
        print("Inputs:", inputs)
        print("Targets:", targets)
        print("Eq positions:", eq_positions)
        #print("eq_token_indexes:", eq_token_indexes)
        print("Mask:", mask)
        break
  
