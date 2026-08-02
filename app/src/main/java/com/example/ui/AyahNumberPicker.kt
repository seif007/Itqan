package com.example.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog

@Composable
fun AyahNumberPicker(
    title: String,
    currentValue: Int,
    maxValue: Int,
    onValueChange: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    var expanded by remember { mutableStateOf(false) }

    OutlinedButton(
        onClick = { expanded = true },
        modifier = modifier.height(32.dp),
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
    ) {
        Text(currentValue.toString(), style = MaterialTheme.typography.bodySmall)
        Icon(Icons.Filled.ArrowDropDown, contentDescription = "Select", modifier = Modifier.size(16.dp))
    }

    if (expanded) {
        Dialog(onDismissRequest = { expanded = false }) {
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = MaterialTheme.colorScheme.surface,
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(0.6f)
            ) {
                Column {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(16.dp)
                    )
                    HorizontalDivider()
                    LazyColumn(
                        modifier = Modifier.weight(1f)
                    ) {
                        items(maxValue) { i ->
                            val ayahNum = i + 1
                            TextButton(
                                onClick = {
                                    onValueChange(ayahNum)
                                    expanded = false
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 16.dp, vertical = 4.dp)
                            ) {
                                Text("Ayah $ayahNum", style = MaterialTheme.typography.bodyLarge)
                            }
                        }
                    }
                    HorizontalDivider()
                    TextButton(
                        onClick = { expanded = false },
                        modifier = Modifier
                            .align(Alignment.End)
                            .padding(8.dp)
                    ) {
                        Text("Cancel")
                    }
                }
            }
        }
    }
}
