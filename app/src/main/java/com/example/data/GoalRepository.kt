package com.example.data

import kotlinx.coroutines.flow.Flow

class GoalRepository(private val goalDao: GoalDao) {
    val allGoals: Flow<List<Goal>> = goalDao.getAllGoals()

    suspend fun insert(goal: Goal) {
        goalDao.insertGoal(goal)
    }

    suspend fun update(goal: Goal) {
        goalDao.updateGoal(goal)
    }

    suspend fun deleteById(id: Int) {
        goalDao.deleteGoalById(id)
    }

    suspend fun deleteAll() {
        goalDao.deleteAllGoals()
    }
}
