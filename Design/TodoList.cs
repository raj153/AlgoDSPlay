using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2590. Design a Todo List
    https://leetcode.com/problems/design-a-todo-list/description/

    Complexity
    •	Time complexity:
        O(1) - Complete task
        O(n) - get all tasks, n is number of tasks per user
        O(nlogn) - get all tasks for tag, n is number of tasks per user, sorting prior return
        O(1) - add tasks for tag, this is more like O(T) where T is number of tags, O(1) is per task connection with tag
    •	Space complexity:
        O(n)

    */
    public class TodoList
    {
        private Dictionary<int, List<Task>> tasksByUserId;
        private int taskSequence;

        public TodoList()
        {
            tasksByUserId = new Dictionary<int, List<Task>>();
            taskSequence = 1;
        }

        public int AddTask(int userId, string taskDescription, int dueDate, List<string> tags)
        {
            if (!tasksByUserId.ContainsKey(userId))
            {
                tasksByUserId[userId] = new List<Task>();
            }

            int taskId = taskSequence++;
            tasksByUserId[userId].Add(new Task(taskId, dueDate, taskDescription, tags));
            return taskId;
        }

        public List<string> GetAllTasks(int userId)
        {
            return GetAllTasks(userId, null);
        }

        public List<string> GetTasksForTag(int userId, string tag)
        {
            return GetAllTasks(userId, tag);
        }

        public void CompleteTask(int userId, int taskId)
        {
            if (tasksByUserId.TryGetValue(userId, out var userTasks))
            {
                foreach (var task in userTasks.Where(task => task.Id == taskId))
                {
                    task.Deleted = true;
                }
            }
        }

        private List<string> GetAllTasks(int userId, string tag)
        {
            if (!tasksByUserId.TryGetValue(userId, out var userTasks))
            {
                return new List<string>();
            }

            return userTasks
                .Where(task => !task.Deleted && (string.IsNullOrEmpty(tag) || task.Tags.Contains(tag)))
                .OrderBy(task => task.DueDate)
                .Select(task => task.Description)
                .ToList();
        }

        private class Task
        {
            public int Id { get; }
            public int DueDate { get; }
            public string Description { get; }
            public bool Deleted { get; set; }
            public HashSet<string> Tags { get; }

            public Task(int id, int dueDate, string description, List<string> tags)
            {
                Id = id;
                DueDate = dueDate;
                Description = description;
                Tags = new HashSet<string>(tags);
                Deleted = false;
            }
        }
    }
    /**
 * Your TodoList object will be instantiated and called as such:
 * TodoList obj = new TodoList();
 * int param_1 = obj.AddTask(userId,taskDescription,dueDate,tags);
 * IList<string> param_2 = obj.GetAllTasks(userId);
 * IList<string> param_3 = obj.GetTasksForTag(userId,tag);
 * obj.CompleteTask(userId,taskId);
 */
}