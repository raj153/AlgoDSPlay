using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class DoublyLinkedList<T1, T2>
    {
        public DoublyLinkedListNode<T1,T2> Head = null;
        public DoublyLinkedListNode<T1,T2> Tail = null;


        // O(1) time | O(1) space
        public void SetHeadTo(DoublyLinkedListNode<T1, T2> node){
            if( Head == node)
                return;
            else if(Head == null){
                Head = node;
                Tail=node;
            }else if(Head == Tail){
                Tail.Prev=node;
                Head = node;
                Head.Next = Tail;
            }else{
                if(Tail == node ){
                    RemoveTail();
                }
                node.RemoveBindings();
                Head.Prev=node;
                node.Next = Head;
                Head = node;
            }
                
        }
        

        public void RemoveTail()
        {
            if(Tail == null)
                return;
            if( Tail == Head)
            {
                Head = null;
                Tail = null;
                return;
                
            }
            Tail = Tail.Prev;
            Tail.Next= null;
        }

        /*
        Non-Generic version
        //https://www.algoexpert.io/questions/linked-list-construction

        public class DoublyLinkedList {
      nodeToInsert.Next = node.Next;
      if (node.Next == null) {
        Tail = nodeToInsert;
      } else {
        node.Next.Prev = nodeToInsert;
      }
      node.Next = nodeToInsert;
    }

    // O(p) time | O(1) space
    public void InsertAtPosition(int position, Node nodeToInsert) {
      if (position == 1) {
        SetHead(nodeToInsert);
        return;
      }
      Node node = Head;
      int currentPosition = 1;
      while (node != null && currentPosition++ != position) node = node.Next;
      if (node != null) {
        InsertBefore(node, nodeToInsert);
      } else {
        SetTail(nodeToInsert);
      }
    }

    // O(n) time | O(1) space
    public void RemoveNodesWithValue(int value) {
      Node node = Head;
      while (node != null) {
        Node nodeToRemove = node;
        node = node.Next;
        if (nodeToRemove.Value == value) Remove(nodeToRemove);
      }
    }

    // O(1) time | O(1) space
    public void Remove(Node node) {
      if (node == Head) Head = Head.Next;
      if (node == Tail) Tail = Tail.Prev;
      RemoveNodeBindings(node);
    }

    // O(n) time | O(1) space
    public bool ContainsNodeWithValue(int value) {
      Node node = Head;
      while (node != null && node.Value != value) node = node.Next;
      return node != null;
    }

    public void RemoveNodeBindings(Node node) {
      if (node.Prev != null) node.Prev.Next = node.Next;
      if (node.Next != null) node.Next.Prev = node.Prev;
      node.Prev = null;
      node.Next = null;
    }
  }

  public class Node {
    public int Value;
    public Node Prev;
    public Node Next;

    public Node(int value) {
      this.Value = value;
    }
  }
        */
    }
}