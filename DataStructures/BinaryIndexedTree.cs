using System;

namespace AlgoDSPlay.DataStructures
{
    public class BinaryIndexedTree
    {
        private int size; // Size of the array
        private int[] tree; // The Binary Indexed Tree (BIT)

        // Constructor
        public BinaryIndexedTree(int size)
        {
            this.size = size;
            this.tree = new int[size + 1];
        }

        // Updates the BIT with a value 'value' at index 'index'
        public void Update(int index, int value)
        {
            while (index <= size)
            {
                tree[index] += value; // Add 'value' to current index
                index += index & -index; // Climb up the tree
            }
        }

        // Queries the cumulative frequency up to index 'index'
        public int Query(int index)
        {
            int sum = 0;
            while (index > 0)
            {
                sum += tree[index]; // Add value at current index to sum
                index -= index & -index; // Climb down the tree
            }
            return sum;
        }
    }
}
