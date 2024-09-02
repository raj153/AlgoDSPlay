using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1206. Design Skiplist
    https://leetcode.com/problems/design-skiplist/description/

    Complexity
    Time complexity:
    •	Search: O(log n)
    •	Insert: O(log n)
    •	Delete: O(log n)
    Space complexity: O(n)
        This space complexity arises from storing the nodes and their forward pointers across multiple levels. The maximum space used is dependent on the number of nodes and the maximum level any node reaches (bounded by MAX_LEVEL).

    */
    public class SkipList
    {
        public readonly int MAX_LEVEL = 20;
        private readonly Node head;
        private int level; // max level the list has
        private readonly Random random; // for randomizing operations

        public SkipList()
        {
            head = new Node(-1, MAX_LEVEL); // initailize head node, head node must be highest level
            level = 0;
            random = new Random();
        }

        // This function create random level for new nodes
        private int RandomLevel()
        {
            int level = 0;
            // There is coin toss logic here - 50%/50%
            // 50% chance of moving to the next level

            while (random.Next(2) == 1 && level < MAX_LEVEL)
            {
                level++; // We must check that we do not exceed the max level limit
            }
            return level;
        }

        public bool Search(int target)
        {
            Node current = head;
            // we start top level and going to bottom level
            for (int i = level; i >= 0; i--)
            {
                while (current.Forward[i] != null && current.Forward[i].Val < target)
                {
                    current = current.Forward[i];
                    // move through nodes that are smaller than the target at the same level 
                }
            }
            current = current.Forward[0];
            return current != null && current.Val == target;
        }

        public void Add(int num)
        {
            // This covers levels from 0 to the max level
            // It is used to update connection on all levels
            Node[] update = new Node[MAX_LEVEL + 1];
            // We dont use [Skiplist.level + 1]. Because levels are randomly create. 
            // So, levels can be up Max_level
            Node current = head;

            // Appropriate places are found by going down from the current levels
            for (int i = level; i >= 0; i--)
            {
                // its like 'temp.next' in linked list.
                while (current.Forward[i] != null && current.Forward[i].Val < num)
                {
                    current = current.Forward[i];
                }
                update[i] = current; // at each level, the node before the node to be added is saved
            }

            int newLevel = RandomLevel();
            if (newLevel > level)
            {
                for (int i = level + 1; i <= newLevel; i++)
                {
                    update[i] = head; // head node is used as reference for new nodes
                }
                level = newLevel;
            }

            Node newNode = new Node(num, newLevel);
            // add new node and update connections
            // update array determines where the new node will be added
            // update[i] keeps the node before the new node at level i
            for (int i = 0; i <= newLevel; i++)
            {
                // the new nodes forward pointer at level i is set to the node pointed to by update[i]s forw. ptr
                newNode.Forward[i] = update[i].Forward[i];
                // The forward pointer of update[i] is updated to point to the new node.
                update[i].Forward[i] = newNode;
            }
        }

        public bool Erase(int num)
        {
            Node[] update = new Node[MAX_LEVEL + 1];
            Node current = head;

            // search for node to delete
            for (int i = level; i >= 0; i--)
            {
                while (current.Forward[i] != null && current.Forward[i].Val < num)
                {
                    current = current.Forward[i];
                }
                update[i] = current;
            }
            current = current.Forward[0];
            // num value is not in Skip List
            if (current == null || current.Val != num) return false;

            // update connections
            for (int i = 0; i <= level; i++)
            {
                if (update[i].Forward[i] != current) break; // we dont have num value
                                                            // We set the forward pointer of update[i] at level i to the next node from the current node
                                                            // Thus, the current node is removed from the Skip List.
                update[i].Forward[i] = current.Forward[i];
            }

            // This loop reduces the level as long as there are empty connections at the top level.
            // This makes Skip List work more efficiently by removing unnecessary levels.
            while (level > 0 && head.Forward[level] == null) { level--; }
            return true;
        }
        public class Node
        {
            public int Val;
            public Node[] Forward; // array of forwards

            public Node(int val, int level)
            {
                this.Val = val;
                Forward = new Node[level + 1];
                // forward is an array that allows its nodes to be connected to each other.
                // we use level + 1 because nodes must has as many pointers as its level
                // note: level is started 0


                // When a create Node object also create a forward array according to nodes level
                // Each node has an array pointer that points to the node that 
                // comes after it in layers up to a certain level.
                // This way, we can quickly make search, add, erase operations
            }
        }


    }
    /**
 * Your Skiplist object will be instantiated and called as such:
 * Skiplist obj = new Skiplist();
 * bool param_1 = obj.Search(target);
 * obj.Add(num);
 * bool param_3 = obj.Erase(num);
 */
}