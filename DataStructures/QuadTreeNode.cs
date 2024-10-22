using System;

namespace AlgoDSPlay.DataStructures
{

    // Definition for a QuadTree node.
    public class QuadTreeNode
    {
        public bool Val;
        public bool IsLeaf;
        public QuadTreeNode TopLeft;
        public QuadTreeNode TopRight;
        public QuadTreeNode BottomLeft;
        public QuadTreeNode BottomRight;

        public QuadTreeNode()
        {
            Val = false;
            IsLeaf = false;
            TopLeft = null;
            TopRight = null;
            BottomLeft = null;
            BottomRight = null;
        }

        public QuadTreeNode(bool _val, bool _isLeaf)
        {
            Val = _val;
            IsLeaf = _isLeaf;
            TopLeft = null;
            TopRight = null;
            BottomLeft = null;
            BottomRight = null;
        }

        public QuadTreeNode(bool _val, bool _isLeaf, QuadTreeNode _topLeft, QuadTreeNode _topRight, QuadTreeNode _bottomLeft, QuadTreeNode _bottomRight)
        {
            Val = _val;
            IsLeaf = _isLeaf;
            TopLeft = _topLeft;
            TopRight = _topRight;
            BottomLeft = _bottomLeft;
            BottomRight = _bottomRight;
        }
    }
}
