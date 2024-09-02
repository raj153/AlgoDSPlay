using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{

  //https://www.algoexpert.io/questions/bst-construction
  public class BST
  {

    public int Value { get; set; }
    public BST Left { get; set; }

    public BST Right { get; set; }

    public BST(int value)
    {
      this.Value = value;
    }
    // Average: O(log(n)) time | O(log(n)) space
    // Worst: O(n) time | O(n) space
    public BST Insert(int value)
    {
      if (value < this.Value)
      {
        if (Left == null)
        {
          BST newBST = new BST(value);
          Left = newBST;
        }
        else
        {
          Left.Insert(value);
        }
      }
      else
      {
        if (Right == null)
        {
          BST newBST = new BST(value);
          Right = newBST;
        }
        else
        {
          Right.Insert(value);
        }
      }
      return this;
    }


    // Average: O(log(n)) time | O(log(n)) space
    // Worst: O(n) time | O(n) space
    public BST Remove(int value)
    {
      Remove(value, null);
      return this;
    }

    public void Remove(int value, BST parent)
    {
      if (value < this.Value)
      {
        if (Left != null)
        {
          Left.Remove(value, this);
        }
      }
      else if (value > this.Value)
      {
        if (Right != null)
        {
          Right.Remove(value, this);
        }
      }
      else
      {
        if (Left != null && Right != null)
        {
          this.Value = Right.getMinValue();
          Right.Remove(this.Value, this);
        }
        else if (parent == null)
        {
          if (Left != null)
          {
            this.Value = Left.Value;
            Right = Left.Right;
            Left = Left.Left;
          }
          else if (Right != null)
          {
            this.Value = Right.Value;
            Left = Right.Left;
            Right = Right.Right;
          }
          else
          {
            // This is a single-node tree; do nothing.
          }
        }
        else if (parent.Left == this)
        {
          parent.Left = Left != null ? Left : Right;
        }
        else if (parent.Right == this)
        {
          parent.Right = Left != null ? Left : Right;
        }

      }
    }

    public int getMinValue()
    {
      if (Left == null)
      {
        return this.Value;
      }
      else
      {
        return Left.getMinValue();
      }
    }

    //https://www.algoexpert.io/questions/bst-traversal
    public static List<int> InOrderTraverse(BST tree, List<int> array)
    {
      //T:O(n)| S:O(n)
      if (tree.Left != null)
      {
        InOrderTraverse(tree.Left, array);
      }
      array.Add(tree.Value);
      if (tree.Right != null)
      {
        InOrderTraverse(tree.Right, array);
      }
      return array;
    }

    public static List<int> PreOrderTraverse(BST tree, List<int> array)
    {
      //T:O(n)| S:O(n)
      array.Add(tree.Value);
      if (tree.Left != null)
      {
        InOrderTraverse(tree.Left, array);
      }
      if (tree.Right != null)
      {
        InOrderTraverse(tree.Right, array);
      }
      return array;
    }

    public static List<int> PostOrderTraverse(BST tree, List<int> array)
    {
      //T:O(n)| S:O(n)

      if (tree.Left != null)
      {
        InOrderTraverse(tree.Left, array);
      }
      if (tree.Right != null)
      {
        InOrderTraverse(tree.Right, array);
      }
      array.Add(tree.Value);
      return array;
    }

  }
}
  /*
  ITERATIVE
    public class BST {
}

// Average: O(log(n)) time | O(1) space
// Worst: O(n) time | O(1) space
public bool Contains(int value) {
BST currentNode = this;
while (currentNode != null) {
  if (value < currentNode.value) {
    currentNode = currentNode.left;
  } else if (value > currentNode.value) {
    currentNode = currentNode.right;
  } else {
    return true;
  }
}
return false;
}

// Average: O(log(n)) time | O(1) space
// Worst: O(n) time | O(1) space
public BST Remove(int value) {
Remove(value, null);
return this;
}

public void Remove(int value, BST parentNode) {
BST currentNode = this;
while (currentNode != null) {
  if (value < currentNode.value) {
    parentNode = currentNode;
    currentNode = currentNode.left;
  } else if (value > currentNode.value) {
    parentNode = currentNode;
    currentNode = currentNode.right;
  } else {
    if (currentNode.left != null && currentNode.right != null) {
      currentNode.value = currentNode.right.getMinValue();
      currentNode.right.Remove(currentNode.value, currentNode);
    } else if (parentNode == null) {
      if (currentNode.left != null) {
        currentNode.value = currentNode.left.value;
        currentNode.right = currentNode.left.right;
        currentNode.left = currentNode.left.left;
      } else if (currentNode.right != null) {
        currentNode.value = currentNode.right.value;
        currentNode.left = currentNode.right.left;
        currentNode.right = currentNode.right.right;
      } else {
        // This is a single-node tree; do nothing.
      }
    } else if (parentNode.left == currentNode) {
      parentNode.left =
        currentNode.left != null ? currentNode.left : currentNode.right;
    } else if (parentNode.right == currentNode) {
      parentNode.right =
        currentNode.left != null ? currentNode.left : currentNode.right;
    }
    break;
  }
}
}

public int getMinValue() {
if (left == null) {
  return value;
} else {
  return left.getMinValue();
}
}
}
  */


