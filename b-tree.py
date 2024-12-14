# Thanks to SK lee who provided the skeleton code

import math, sys
import pandas as pd
import numpy as np
from tqdm import tqdm


class Node:
    def __init__(self, leaf=False):
        self.keys = []
        self.parent = []
        self.children = []
        self.leaf = leaf


class BTree:
    def __init__(self, t):
        """
        # create a instance of the Class of a B-Tree
        # t : the minimum degree t
        # (the max num of keys is 2*t -1, the min num of keys is t-1)
        """
        self.root = Node(True)
        self.t = t

    # B-Tree-Split-Child
    def split_child(self, x, i):
        """
        # split the node x's i-th child that is full
        # x: the current node
        # i: the index of the node x's child to be split
        # return: None
        """
        y = x.children[i]
        z = Node(y.leaf)  # Create a new node z.

        # Move middle key of y to x and insert z as a new child.
        x.keys.insert(i, y.keys[self.t - 1])
        x.children.insert(i + 1, z)
        z.parent = x

        # Split y's keys and children.
        z.keys = y.keys[self.t :]
        y.keys = y.keys[: self.t - 1]

        if not y.leaf:
            z.children = y.children[self.t :]
            y.children = y.children[: self.t]
            for child in z.children:
                child.parent = z

    # B-Tree-Insert
    def insert(self, k):
        """
        # insert the key k into the B-Tree
        # return: None
        """
        root = self.root

        # If root is full, split it.
        if len(root.keys) == (2 * self.t) - 1:
            new_root = Node()
            new_root.children.append(root)
            root.parent = new_root
            self.split_child(new_root, 0)
            self.root = new_root

        self.insert_key(self.root, k)  # Insert into the non-full root.

    # B-Tree-Insert-Nonfull
    def insert_key(self, x, k):
        """
        # insert the key k into node x
        # return: None
        """
        if x.leaf:
            x.keys.append(k)
            x.keys.sort()
        else:
            # Find child to insert into.
            i = len(x.keys) - 1
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1

            # Split child if full.
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                # Determine which of the two children to insert into.
                if k > x.keys[i]:
                    i += 1

            self.insert_key(x.children[i], k)  # Recur to insert the key.

    # B-Tree-Search
    def search_key(self, x, key):
        """
        # search for the key in node x
        # return: the node x that contains the key, the index of the key if the key is in the B-tree
        """
        i = 0
        while i < len(x.keys) and key > x.keys[i][0]:
            i += 1

        if i < len(x.keys) and key == x.keys[i][0]:  # Key found.
            return (x, i)

        if x.leaf:  # Key not found.
            return (None, None)

        return self.search_key(x.children[i], key)  # Recur into the child.

    def delete(self, k):
        """
        # delete the key k from the B-tree
        # return: None
        """
        node, idx = self.search_key(self.root, k)  # Find node and index of key.

        if node is None:  # Key not found.
            return

        if node.leaf:
            self.delete_leaf_node(node, idx)
        else:
            self.delete_internal_node(node, idx)

    def delete_leaf_node(self, x, i):
        """
        # delete the key at index i from the leaf node x.
        return: None
        """
        # If the node is the root, remove the key and update the root if empty.
        if x.parent is None:
            x.keys.pop(i)
            if len(x.keys) == 0:
                self.root = None
            return

        # If the node has enough keys, remove the key.
        if len(x.keys) >= self.t:
            x.keys.pop(i)
        else:
            parent = x.parent
            idx = parent.children.index(x)

            # Try borrowing from left sibling.
            if idx > 0 and len(parent.children[idx - 1].keys) >= self.t:
                x.keys.pop(i)
                self.borrow_from_left(x, idx - 1)

            # Try borrowing from right sibling.
            elif (
                idx < len(parent.children) - 1
                and len(parent.children[idx + 1].keys) >= self.t
            ):
                x.keys.pop(i)
                self.borrow_from_right(x, idx + 1)

            # Merge with a sibling if neither can lend a key.
            else:
                key_to_delete = x.keys[i][0]
                if idx == 0:
                    self.merge_sibling(parent, idx, idx + 1)
                else:
                    self.merge_sibling(parent, idx - 1, idx)

                self.fixup(parent)  # Fix the parent after merging.
                self.delete(key_to_delete)  # Recur to delete the key.

    def borrow_from_left(self, x, idx):
        """
        # borrow a key from the left sibling of node x
        # return: None
        """
        left_sibling = x.parent.children[idx]
        x.keys.insert(0, x.parent.keys[idx])  # Move parent's key to x.
        x.parent.keys[idx] = (
            left_sibling.keys.pop()
        )  # Replace parent's key with left sibling's key.

        if not x.leaf:
            x.children.insert(
                0, left_sibling.children.pop()
            )  # Move last child from left sibling to x.
            x.children[0].parent = x  # Update child's parent.

    def borrow_from_right(self, x, idx):
        """
        # borrow a key from the right sibling of node x
        # return: None
        """
        right_sibling = x.parent.children[idx]
        x.keys.append(x.parent.keys[idx - 1])  # Move parent's key to x.
        x.parent.keys[idx - 1] = right_sibling.keys.pop(
            0
        )  # Replace parent's key with right sibling's key.

        if not x.leaf:
            x.children.append(
                right_sibling.children.pop(0)
            )  # Move first child from right sibling to x.
            x.children[-1].parent = x  # Update child's parent.

    def merge_sibling(self, x, i, j):
        """
        # merge sibling j into sibling i
        # return: None
        """
        left_child = x.children[i]
        right_child = x.children[j]

        left_child.keys.append(x.keys.pop(i))  # Move parent's key to left child.
        left_child.keys.extend(
            right_child.keys
        )  # Transfer all keys from right child to left child.

        # If not a leaf, transfer the children and update the parent references.
        if not left_child.leaf:
            left_child.children.extend(right_child.children)
            for child in left_child.children:
                child.parent = left_child

        x.children.pop(j)  # Remove the right child from the parent's child list.

    def fixup(self, x):
        """
        # recursively fix the parent node after merging or borrowing
        # x: the node that needs to be fixed
        # return: None
        """
        if x == self.root:
            # Update root if it has no keys.
            if len(x.keys) == 0:
                if len(x.children) == 0:  # Tree is empty.
                    self.root = None
                else:  # Update root to the first child.
                    self.root = x.children[0]
                    self.root.parent = None
            return

        # If the node has enough keys, no further action is needed.
        if len(x.keys) >= self.t - 1:
            return

        parent = x.parent
        idx = parent.children.index(x)

        # Try borrowing from left sibling.
        if idx > 0 and len(parent.children[idx - 1].keys) >= self.t:
            self.borrow_from_left(x, idx - 1)

        # Otherwise, try borrowing from right sibling.
        elif (
            idx < len(parent.children) - 1
            and len(parent.children[idx + 1].keys) >= self.t
        ):
            self.borrow_from_right(x, idx + 1)

        # If borrowing isn't possible, merge with a sibling and recursively fix the parent.
        else:
            if idx == 0:
                self.merge_sibling(parent, idx, idx + 1)
            else:
                self.merge_sibling(parent, idx - 1, idx)
            self.fixup(parent)

    def delete_internal_node(self, x, i):
        """
        # delete the key at index i in an internal node x
        # return: None
        """
        key_to_delete = x.keys[i][0]
        left_child = x.children[i]
        right_child = x.children[i + 1]

        # If the left child has enough keys, replace the key with its predecessor and delete the key.
        if len(left_child.keys) >= self.t:
            predecessor = self.find_predecessor(left_child)
            x.keys[i] = predecessor.keys[-1]
            self.delete_leaf_node(predecessor, len(predecessor.keys) - 1)

        # If the right child has enough keys, replace the key with its successor and delete the key.
        elif len(right_child.keys) >= self.t:
            successor = self.find_successor(right_child)
            x.keys[i] = successor.keys[0]
            self.delete_leaf_node(successor, 0)

        # If neither child has enough keys, merge the two children and delete the key.
        else:
            self.merge_sibling(x, i, i + 1)
            self.fixup(x)
            self.delete(key_to_delete)

    def find_predecessor(self, x):
        """
        # find the predecessor of the key in the subtree rooted at node x
        # return: the node containing the predecessor key
        """
        current = x
        while not current.leaf:
            current = current.children[-1]
        return current

    def find_successor(self, x):
        """
        # find the successor of the key in the subtree rooted at node x
        # return: the node containing the successor key
        """
        current = x
        while not current.leaf:
            current = current.children[0]
        return current

    # for printing the statistic of the resulting B-tree
    def traverse_key(self, x, level=0, level_counts=None):
        """
        # run BFS on the B-tree to count the number of keys at every level
        # return: level_counts
        """
        if level_counts is None:
            level_counts = {}

        if x:
            # counting the number of keys at the current level
            if level in level_counts:
                level_counts[level] += len(x.keys)
            else:
                level_counts[level] = len(x.keys)

            # recursively call the traverse_key() for further traverse
            for child in x.children:
                self.traverse_key(child, level + 1, level_counts)

        return level_counts


# Btree Class done


def get_file():
    """
    # read an input file (.csv) with its name
    """
    file_name = input(
        "Enter the file name you want to insert or delete ▷ (e.g., insert1 or delete1_50 or delete1_90 or ...) "
    )

    while True:
        try:
            file = pd.read_csv(
                "inputs/" + file_name + ".csv", delimiter="\t", names=["key", "value"]
            )
            return file
        except FileNotFoundError:
            print("File does not exist.")
            file_name = input("Enter the file name again. ▷ ")


def insertion_test(B, file):
    """
    #   read all keys and values from the file and insert them into the B-tree
    #   B   : an empty B-tree
    #   file: a csv file that contains keys to be inserted
    #   return: the resulting B-tree
    """

    file_key = file["key"]
    file_value = file["value"]

    print("===============================")
    print("[ Insertion start ]")

    for i in tqdm(
        range(len(file_key))
    ):  # tqdm shows the insertion progress and the elapsed time
        B.insert([file_key[i], file_value[i]])

    print("[ Insertion complete ]")
    print("===============================")
    print()

    return B


def deletion_test(B, delete_file):
    """
    #   read all keys and values from the file and delete them from the B-tree
    #   B   : the current B-tree
    #   file: a csv file that contains keys to be deleted
    #   return: the resulting B-tree
    """

    delete_key = delete_file["key"]

    print("===============================")
    print("[ Deletion start ]")

    for i in tqdm(range(len(delete_key))):
        B.delete(delete_key[i])

    print("[ Deletion complete ]")
    print("===============================")
    print()

    return B


def print_statistic(B):
    """
    # print the information about the current B-tree
    # the number of keys at each level
    # the total number of keys in the B-tree
    """
    print("===============================")
    print("[ Print statistic of tree ]")

    level_counts = B.traverse_key(B.root)

    for level, counts in level_counts.items():
        if level == 0:
            print(f"Level {level} (root): Key Count = {counts}")
        else:
            print(f"Level {level}: Key Count = {counts}")
    print("-------------------------------")
    total_keys = sum(counts for counts in level_counts.values())
    print(f"Total number of keys across all levels: {total_keys}")
    print("[ Print complete ]")
    print("===============================")
    print()


def main():
    B = None
    while True:
        try:
            num = int(input("1.insertion 2.deletion. 3.statistic 4.end ▶  "))

            # 1. Insertion
            if num == 1:
                t = 3  # minimum degree
                B = BTree(t)  # make an empty b-tree with the minimum degree t
                insert_file = get_file()
                B = insertion_test(B, insert_file)

            # 2. Deletion
            elif num == 2:
                delete_file = get_file()
                B = deletion_test(B, delete_file)

            # 3. Statistic
            elif num == 3:
                print_statistic(B)

            # 4. End program
            elif num == 4:
                sys.exit(1)

            else:
                print("Invalid input. Please enter 1, 2, 3, or 4.")

        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    main()
