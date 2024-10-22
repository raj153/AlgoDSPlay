using System;

namespace AlgoDSPlay.Design
{

    /* 1993. Operations on Tree
    https://leetcode.com/problems/operations-on-tree/description/\
Complexity
•	Time complexity: O(n)
o	Constructor: O(n), due to children initialization.
o	Lock: O(1)
o	Unlock: O(1)
o	Upgrade: O(n), due to possibility of full tree traversal (Upgrade root).
•	Space complexity: O(n)
o	Parent array: O(n)
o	Locked array: O(n)
o	Children array: O(n + n), due to need to add almost every node as a child.

     */
    public class LockingTreeSol
    {
        class LockingTreeUsingDict
        {
            private int[] parent;
            private List<int>[] tree;
            private Dictionary<int, int> locked; // node, user.

            public LockingTreeUsingDict (int[] parent)
            {
                int n = parent.Length;
                tree = new List<int>[n];
                locked = new Dictionary<int, int>();
                this.parent = parent;

                for (int i = 0; i < n; i++)
                    tree[i] = new List<int>();
                for (int i = 1; i < n; i++)
                    tree[parent[i]].Add(i);
            }

            public bool Lock(int num, int user)
            {
                if (locked.ContainsKey(num)) return false;
                locked[num] = user;
                return true;
            }

            public bool Unlock(int num, int user)
            {
                if (!locked.ContainsKey(num) || locked[num] != user) return false;
                locked.Remove(num);
                return true;
            }

            public bool Upgrade(int num, int user)
            {
                if (locked.ContainsKey(num)) return false;

                // check if all the ancestor nodes are unlocked.
                int curr = num;
                while (curr != -1)
                {
                    curr = parent[curr];
                    if (locked.ContainsKey(curr))
                        return false;
                }

                // check if num has at least one locked descendant
                int tmp = locked.Count;
                Dfs(num);
                if (tmp == locked.Count) return false;

                locked[num] = user;
                return true;
            }

            public void Dfs(int src)
            {
                if (locked.ContainsKey(src))
                    locked.Remove(src);
                foreach (int nbr in tree[src])
                    Dfs(nbr);
            }
        }
        public class LockingTreeUsingArray
        {
            private int[] _parentTree; // Stores initial parent index tree.
            private int[] _locks; // Tracks the current user locking a given node.
            private List<int>[] _children; // Stores children indices for a given node.

            private const int _unlocked = 0; // Unlock value. Users have range [1 - 10e4].

            public LockingTreeUsingArray(int[] parent)
            {
                _parentTree = parent;
                _locks = new int[parent.Length];
                AssignChildren();
            }

            public bool Lock(int num, int user)
            {
                if (_locks[num] != _unlocked) { return false; }
                _locks[num] = user;
                return true;
            }

            public bool Unlock(int num, int user)
            {
                if (_locks[num] != user) { return false; }
                _locks[num] = _unlocked;
                return true;
            }

            public bool Upgrade(int num, int user)
            {
                if (SelfOrParentsAreLocked(num) || UnlockChildren(num)) { return false; }
                _locks[num] = user;
                return true;
            }

            // Create storage for child indices. Assign the children for each node as necessary.
            private void AssignChildren()
            {
                // Create an array for each node's children, which will initially be null.
                _children = new List<int>[_parentTree.Length];

                // Index 0 will always be the root (-1), so it can be skipped.
                for (var index = 1; index < _parentTree.Length; ++index)
                {
                    var parent = _children[_parentTree[index]] ??= new();
                    parent.Add(index);
                }
            }

            // Work up the parent tree, until the root has been reached (parent is -1).
            private bool SelfOrParentsAreLocked(int current)
            {
                for (; current >= 0; current = _parentTree[current])
                {
                    if (_locks[current] != _unlocked) { return true; }
                }
                return false;
            }

            // !!! Must be called after the Upgrade has passed the parent and ancestor checks. !!!
            // Unlocks all children, tracking if at least one has been upgraded.
            private bool UnlockChildren(int parent)
            {
                var queue = new Queue<int>(); // Use queue to perform a BFS of all the children nodes.
                queue.Enqueue(parent);

                var noUnlocks = true;
                while (queue.Count > 0)
                {
                    var node = queue.Dequeue();
                    if (_locks[node] != _unlocked) // Unlock and acknowledge lock change for child.
                    {
                        noUnlocks = false;
                        _locks[node] = _unlocked;
                    }

                    if (_children[node] == null) { continue; } // No children are present so continue.
                    foreach (var child in _children[node]) { queue.Enqueue(child); } // Add children to queue.
                }
                return noUnlocks;
            }

        }
        /**
     * Your LockingTree object will be instantiated and called as such:
     * LockingTree obj = new LockingTree(parent);
     * bool param_1 = obj.Lock(num,user);
     * bool param_2 = obj.Unlock(num,user);
     * bool param_3 = obj.Upgrade(num,user);
     */
    }
}
