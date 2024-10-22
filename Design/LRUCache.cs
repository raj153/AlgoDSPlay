using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    146. LRU Cache
    https://leetcode.com/problems/lru-cache/description/

    Approach 1: Doubly Linked List

    Time complexity: O(1) for both get and put.
    For get: Check if a key is in a hash map. This costs O(1). Get a node associated with a key. This costs O(1). Call remove and add. Both methods cost O(1).
    For put: Check if a key is in a hash map. This costs O(1). If it is, we get a node associated with a key and call remove. Both cost O(1). Create a new node and insert it into the hash map. This costs O(1). Call add. This method costs O(1).
                If the capacity is exceeded, we call remove and delete from the hash map. Both cost O(1).
    Space complexity: O(capacity) We use extra space for the hash map and for our linked list. Both cannot exceed a size of capacity.
    */
    public class LRUCache
    {

        private int capacity;
        private Dictionary<int, Node> dic;
        private Node head;
        private Node tail;

        public LRUCache(int capacity)
        {
            this.capacity = capacity;
            dic = new Dictionary<int, Node>();
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.Next = tail;
            tail.Prev = head;
        }

        public int Get(int key)
        {
            if (!dic.ContainsKey(key))
            {
                return -1;
            }

            Node node = dic[key];
            Remove(node);
            Add(node);
            return node.Val;
        }

        public void Put(int key, int value)
        {
            if (dic.ContainsKey(key))
            {
                Node oldNode = dic[key];
                Remove(oldNode);
            }

            Node node = new Node(key, value);
            dic[key] = node;
            Add(node);
            if (dic.Count > capacity)
            {
                Node nodeToDelete = head.Next;
                Remove(nodeToDelete);
                dic.Remove(nodeToDelete.Key);
            }
        }

        private void Add(Node node)
        {
            Node previousEnd = tail.Prev;
            previousEnd.Next = node;
            node.Prev = previousEnd;
            node.Next = tail;
            tail.Prev = node;
        }

        private void Remove(Node node)
        {
            node.Prev.Next = node.Next;
            node.Next.Prev = node.Prev;
        }
    }
    public class Node
    {

        public int Key { get; set; }
        public int Val { get; set; }
        public Node Next { get; set; }
        public Node Prev { get; set; }

        public Node(int key, int value)
        {
            this.Key = key;
            this.Val = value;
        }

        public Node(int value1, Node prev, Node next)
        {
            this.Val = value1;
            this.Prev = prev;
            this.Next = next;
        }
    }

    
    //Approach 2: Built-in data structures
    /*
    •	Time complexity: O(1) for both get and put.
    •	Space complexity: O(capacity) 
    */
    public class LRUCache2
    {
        private int capacity;
        private Dictionary<int, LinkedListNode<TwoInt>> dic;

        private LinkedList<TwoInt> lru;

        // Helper class
        private class TwoInt
        {
            public int Key { get; set; }
            public int Value { get; set; }

            public TwoInt(int key, int val)
            {
                Key = key;
                Value = val;
            }
        }

        public LRUCache2(int capacity)
        {
            this.capacity = capacity;
            dic = new Dictionary<int, LinkedListNode<TwoInt>>();
            lru = new LinkedList<TwoInt>();
        }

        public int Get(int key)
        {
            LinkedListNode<TwoInt> node;
            if (dic.TryGetValue(key, out node))
            {
                // Move to front
                var value = node.Value.Value;
                lru.Remove(node);
                dic[key] = new LinkedListNode<TwoInt>(new TwoInt(key, value));
                lru.AddFirst(dic[key]);
                return value;
            }
            else
            {
                return -1;
            }
        }

        public void Put(int key, int value)
        {
            if (dic.ContainsKey(key))
            {
                // Exist
                lru.Remove(dic[key]);
                dic.Remove(key);
            }

            dic[key] = new LinkedListNode<TwoInt>(new TwoInt(key, value));
            lru.AddFirst(dic[key]);
            // Check capacity
            if (dic.Count > capacity)
            {
                int lastKey = lru.Last.Value.Key;
                lru.RemoveLast();
                dic.Remove(lastKey);
            }
        }
    }
    /**
     * Your LRUCache object will be instantiated and called as such:
     * LRUCache obj = new LRUCache(capacity);
     * int param_1 = obj.Get(key);
     * obj.Put(key,value);
     */

}
