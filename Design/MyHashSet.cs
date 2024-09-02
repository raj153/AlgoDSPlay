using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    705. Design HashSet
    https://leetcode.com/problems/design-hashset/description/

    */
    public class MyHashSet
    {
        /*
        Approach 1: LinkedList as Bucket (LLB)

        Complexity Analysis
        •	Time Complexity: O(KN) where N is the number of all possible values and K is the number of predefined buckets, which is 769.
            o	Assuming that the values are evenly distributed, thus we could consider that the average size of bucket is KN.
            o	Since for each operation, in the worst case, we would need to scan the entire bucket, hence the time complexity is O(KN).
        •	Space Complexity: O(K+M) where K is the number of predefined buckets, and M is the number of unique values that have been inserted into the HashSet.

        */
        public class MyHashSetLLB
        {
            private Bucket[] bucketArray;
            private int keyRange;

            /** Initialize your data structure here. */
            public MyHashSetLLB()
            {
                this.keyRange = 769;
                this.bucketArray = new Bucket[this.keyRange];
                for (int i = 0; i < this.keyRange; ++i)
                    this.bucketArray[i] = new Bucket();
            }

            protected int Hash(int key)
            {
                return (key % this.keyRange);
            }

            public void Add(int key)
            {
                int bucketIndex = this.Hash(key);
                this.bucketArray[bucketIndex].Insert(key);
            }

            public void Remove(int key)
            {
                int bucketIndex = this.Hash(key);
                this.bucketArray[bucketIndex].Delete(key);
            }

            /** Returns true if this set contains the specified element */
            public bool Contains(int key)
            {
                int bucketIndex = this.Hash(key);
                return this.bucketArray[bucketIndex].Exists(key);
            }

            public class Bucket
            {
                private LinkedList<int> container;

                public Bucket()
                {
                    container = new LinkedList<int>();
                }

                public void Insert(int key)
                {
                    if (!container.Contains(key))
                    {
                        container.AddFirst(key);
                    }
                }

                public void Delete(int key)
                {
                    container.Remove(key);
                }

                public bool Exists(int key)
                {
                    return container.Contains(key);
                }
            }
        }

        /*
        Approach 2: Binary Search Tree (BST) as Bucket (BSTB)

        Complexity Analysis
        •	Time Complexity: O(logKN) where N is the number of all possible values and K is the number of predefined buckets, which is 769.
            o	Assuming that the values are evenly distributed, we could consider that the average size of bucket is KN.
            o	When we traverse the BST, we are conducting binary search, as a result, the final time complexity of each operation is O(logKN).
        •	Space Complexity: O(K+M) where K is the number of predefined buckets, and M is the number of unique values that have been inserted into the HashSet.

        */
        public class MyHashSetBSTB
        {
            private Bucket[] bucketArray;
            private int keyRange;

            /** Initialize your data structure here. */
            public MyHashSetBSTB()
            {
                this.keyRange = 769;
                this.bucketArray = new Bucket[this.keyRange];
                for (int i = 0; i < this.keyRange; ++i)
                    this.bucketArray[i] = new Bucket();
            }

            protected int Hash(int key)
            {
                return (key % this.keyRange);
            }

            public void Add(int key)
            {
                int bucketIndex = this.Hash(key);
                this.bucketArray[bucketIndex].Insert(key);
            }

            public void Remove(int key)
            {
                int bucketIndex = this.Hash(key);
                this.bucketArray[bucketIndex].Delete(key);
            }

            /** Returns true if this set contains the specified element */
            public bool Contains(int key)
            {
                int bucketIndex = this.Hash(key);
                return this.bucketArray[bucketIndex].Exists(key);
            }

            public class Bucket
            {
                private BSTree tree;

                public Bucket()
                {
                    tree = new BSTree();
                }

                public void Insert(int key)
                {
                    this.tree.root = this.tree.InsertIntoBST(this.tree.root, key);
                }

                public void Delete(int key)
                {
                    this.tree.root = this.tree.DeleteNode(this.tree.root, key);
                }

                public bool Exists(int key)
                {
                    TreeNode node = this.tree.SearchBST(this.tree.root, key);
                    return (node != null);
                }
            }

            public class TreeNode
            {
                public int Value;
                public TreeNode Left;
                public TreeNode Right;

                public TreeNode(int x)
                {
                    Value = x;
                }
            }

            public class BSTree
            {
                public TreeNode root = null;

                public TreeNode SearchBST(TreeNode root, int val)
                {
                    if (root == null || val == root.Value)
                        return root;

                    return val < root.Value ? SearchBST(root.Left, val) : SearchBST(root.Right, val);
                }

                public TreeNode InsertIntoBST(TreeNode root, int val)
                {
                    if (root == null)
                        return new TreeNode(val);

                    if (val > root.Value)
                        // insert into the right subtree
                        root.Right = InsertIntoBST(root.Right, val);
                    else if (val == root.Value)
                        // skip the insertion
                        return root;
                    else
                        // insert into the left subtree
                        root.Left = InsertIntoBST(root.Left, val);
                    return root;
                }

                /*
                 * One step right and then always left
                 */
                public int Successor(TreeNode root)
                {
                    root = root.Right;
                    while (root.Left != null)
                        root = root.Left;
                    return root.Value;
                }

                /*
                 * One step left and then always right
                 */
                public int Predecessor(TreeNode root)
                {
                    root = root.Left;
                    while (root.Right != null)
                        root = root.Right;
                    return root.Value;
                }

                public TreeNode DeleteNode(TreeNode root, int key)
                {
                    if (root == null)
                        return null;

                    // delete from the right subtree
                    if (key > root.Value)
                        root.Right = DeleteNode(root.Right, key);
                    // delete from the left subtree
                    else if (key < root.Value)
                        root.Left = DeleteNode(root.Left, key);
                    // delete the current node
                    else
                    {
                        // the node is a leaf
                        if (root.Left == null && root.Right == null)
                            root = null;
                        // the node is not a leaf and has a right child
                        else if (root.Right != null)
                        {
                            root.Value = Successor(root);
                            root.Right = DeleteNode(root.Right, root.Value);
                        }
                        // the node is not a leaf, has no right child, and has a left child
                        else
                        {
                            root.Value = Predecessor(root);
                            root.Left = DeleteNode(root.Left, root.Value);
                        }
                    }
                    return root;
                }
            }
        }


    }
    /**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet obj = new MyHashSet();
 * obj.Add(key);
 * obj.Remove(key);
 * bool param_3 = obj.Contains(key);
 */
}