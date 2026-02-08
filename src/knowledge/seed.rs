use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::memory::store::MiKnowledge;

/// Loads MI principles from the knowledge base and flattens into embeddable chunks.
///
/// Parses `mi_principles.json` from the given directory and creates one `MiKnowledge`
/// chunk per concept (MI spirit component, OARS skill, change talk type, etc.).
/// Each chunk is self-contained with enough context for meaningful vector retrieval.
#[tracing::instrument(level = "info")]
pub fn load_mi_principles(kb_path: &Path) -> Result<Vec<MiKnowledge>> {
    let file_path = kb_path.join("mi_principles.json");
    let file = std::fs::File::open(&file_path)
        .with_context(|| format!("Failed to open {}", file_path.display()))?;
    let json: Value = serde_json::from_reader(file).context("Failed to parse mi_principles.json")?;

    let mut chunks = Vec::new();

    // MI Spirit components
    if let Some(components) = json.pointer("/mi_spirit/components").and_then(Value::as_object) {
        for (name, component) in components {
            let mut content = format!("MI Spirit - {}\n", titlecase(name));
            append_str(&mut content, component, "description");
            if let Some(peer) = component.get("peer_adapted").and_then(Value::as_str) {
                content.push_str(&format!("Peer approach: \"{peer}\"\n"));
            }
            if let Some(avoid) = component.get("what_to_avoid").and_then(Value::as_str) {
                content.push_str(&format!("Avoid: {avoid}\n"));
            }
            chunks.push(MiKnowledge {
                id: format!("mi_spirit_{name}"),
                category: "mi_spirit".into(),
                topic: name.clone(),
                content,
            });
        }
    }

    // OARS skills
    if let Some(oars) = json.get("oars").and_then(Value::as_object) {
        for skill_name in &["open_questions", "affirmations", "reflections", "summaries"] {
            if let Some(skill) = oars.get(*skill_name) {
                let mut content = format!("OARS - {}\n", titlecase(skill_name));
                append_str(&mut content, skill, "description");
                append_str(&mut content, skill, "purpose");
                append_str_array(&mut content, skill, "when_to_use", "When to use");
                append_str_array(&mut content, skill, "peer_adapted", "Peer examples");
                append_str_array(&mut content, skill, "what_to_avoid", "Avoid");

                // Handle reflection sub-types
                for sub in &[
                    "simple_reflections",
                    "complex_reflections",
                    "amplified_reflections",
                    "double_sided_reflections",
                ] {
                    if let Some(sub_skill) = skill.get(*sub) {
                        content.push_str(&format!("\n{}: ", titlecase(sub)));
                        append_str(&mut content, sub_skill, "description");
                        append_str_array(&mut content, sub_skill, "peer_adapted", "Examples");
                    }
                }

                // Handle summary types
                if let Some(types) = skill.pointer("/types").and_then(Value::as_object) {
                    for (type_name, type_val) in types {
                        content.push_str(&format!("\n{}: ", titlecase(type_name)));
                        append_str(&mut content, type_val, "description");
                        append_str(&mut content, type_val, "when_to_use");
                    }
                }

                append_str_array(
                    &mut content,
                    skill,
                    "peer_adapted_examples",
                    "Peer examples",
                );
                append_str_array(&mut content, skill, "reflection_tips", "Tips");

                chunks.push(MiKnowledge {
                    id: format!("oars_{skill_name}"),
                    category: "oars".into(),
                    topic: skill_name.to_string(),
                    content,
                });
            }
        }
    }

    // Change talk (DARN-CAT)
    if let Some(ct) = json.get("change_talk") {
        // Preparatory change talk
        if let Some(prep) = ct.get("preparatory_change_talk").and_then(Value::as_object) {
            for talk_type in &["desire", "ability", "reasons", "need"] {
                if let Some(val) = prep.get(*talk_type) {
                    let mut content =
                        format!("Change Talk - Preparatory - {} (DARN)\n", titlecase(talk_type));
                    append_str(&mut content, val, "description");
                    append_str_array(&mut content, val, "examples", "Client examples");
                    append_str_array(&mut content, val, "peer_responses", "Peer responses");
                    chunks.push(MiKnowledge {
                        id: format!("change_talk_prep_{talk_type}"),
                        category: "change_talk".into(),
                        topic: format!("preparatory_{talk_type}"),
                        content,
                    });
                }
            }
        }

        // Mobilizing change talk
        if let Some(mob) = ct.get("mobilizing_change_talk").and_then(Value::as_object) {
            for talk_type in &["commitment", "activation", "taking_steps"] {
                if let Some(val) = mob.get(*talk_type) {
                    let mut content = format!(
                        "Change Talk - Mobilizing - {} (CAT)\n",
                        titlecase(talk_type)
                    );
                    append_str(&mut content, val, "description");
                    append_str_array(&mut content, val, "examples", "Client examples");
                    append_str_array(&mut content, val, "peer_responses", "Peer responses");
                    chunks.push(MiKnowledge {
                        id: format!("change_talk_mob_{talk_type}"),
                        category: "change_talk".into(),
                        topic: format!("mobilizing_{talk_type}"),
                        content,
                    });
                }
            }
        }

        // Evocation strategies
        let mut evoke_content = "Evoking Change Talk - Strategies\n".to_string();
        append_str_array(&mut evoke_content, ct, "how_to_evoke_change_talk", "Strategies");
        append_str(&mut evoke_content, ct, "recognizing_change_talk");
        if evoke_content.len() > 50 {
            chunks.push(MiKnowledge {
                id: "change_talk_evocation".into(),
                category: "change_talk".into(),
                topic: "evocation_strategies".into(),
                content: evoke_content,
            });
        }
    }

    // Sustain talk and discord
    if let Some(sd) = json.get("sustain_talk_and_discord") {
        // Sustain talk
        if let Some(st) = sd.get("sustain_talk") {
            let mut content = "Sustain Talk - Language Favoring Status Quo\n".to_string();
            append_str(&mut content, st, "description");
            append_str_array(&mut content, st, "examples", "Examples");
            append_str_array(&mut content, st, "how_to_respond", "How to respond");
            append_str_array(&mut content, st, "peer_adapted_responses", "Peer responses");
            chunks.push(MiKnowledge {
                id: "sustain_talk".into(),
                category: "sustain_talk".into(),
                topic: "sustain_talk".into(),
                content,
            });
        }

        // Discord
        if let Some(discord) = sd.get("discord") {
            let mut content = "Discord - Dissonance in the Helping Relationship\n".to_string();
            append_str(&mut content, discord, "description");
            append_str_array(&mut content, discord, "signs_of_discord", "Signs");
            append_str(&mut content, discord, "discord_as_feedback");

            // Discord strategies
            if let Some(strategies) =
                discord.get("strategies_for_discord").and_then(Value::as_object)
            {
                for (strat_name, strat_val) in strategies {
                    content.push_str(&format!("\nStrategy - {}: ", titlecase(strat_name)));
                    append_str(&mut content, strat_val, "description");
                    append_str_array(&mut content, strat_val, "examples", "Examples");
                    append_str_array(&mut content, strat_val, "peer_examples", "Peer examples");
                }
            }

            chunks.push(MiKnowledge {
                id: "discord".into(),
                category: "sustain_talk".into(),
                topic: "discord".into(),
                content,
            });
        }
    }

    // Asking permission
    if let Some(ap) = json.get("asking_permission") {
        let mut content = "Asking Permission Before Giving Advice\n".to_string();
        append_str(&mut content, ap, "description");
        append_str(&mut content, ap, "why");
        append_str_array(&mut content, ap, "peer_adapted", "Peer examples");
        chunks.push(MiKnowledge {
            id: "asking_permission".into(),
            category: "technique".into(),
            topic: "asking_permission".into(),
            content,
        });
    }

    // Peer boundaries
    if let Some(pb) = json.get("peer_boundaries") {
        // Scope
        let mut content = "Peer Support Boundaries - Scope\n".to_string();
        append_str_array(&mut content, pb, "in_scope", "In scope");
        append_str_array(&mut content, pb, "out_of_scope", "Out of scope");
        append_str_array(&mut content, pb, "when_to_refer", "When to refer");
        chunks.push(MiKnowledge {
            id: "peer_boundaries_scope".into(),
            category: "peer_boundaries".into(),
            topic: "scope".into(),
            content,
        });

        // Crisis protocols
        if let Some(protocols) = pb.get("crisis_protocols").and_then(Value::as_object) {
            for (crisis_type, protocol) in protocols {
                let mut content = format!("Crisis Protocol - {}\n", titlecase(crisis_type));
                append_str_array(&mut content, protocol, "indicators", "Warning signs");
                if let Some(template) =
                    protocol.get("peer_response_template").and_then(Value::as_str)
                {
                    content.push_str(&format!("Response: {template}\n"));
                }
                append_str_array(&mut content, protocol, "what_not_to_do", "Do NOT");
                chunks.push(MiKnowledge {
                    id: format!("crisis_{crisis_type}"),
                    category: "crisis".into(),
                    topic: crisis_type.clone(),
                    content,
                });
            }
        }

        // Resources
        if let Some(resources) = pb.get("resources_to_know").and_then(Value::as_object) {
            let mut content = "Crisis and Mental Health Resources\n".to_string();
            for (category, items) in resources {
                content.push_str(&format!("\n{}:\n", titlecase(category)));
                if let Some(arr) = items.as_array() {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            content.push_str(&format!("- {s}\n"));
                        }
                    }
                }
            }
            chunks.push(MiKnowledge {
                id: "resources".into(),
                category: "crisis".into(),
                topic: "resources".into(),
                content,
            });
        }
    }

    // Common mistakes
    if let Some(mistakes) = json.get("common_mistakes_to_avoid").and_then(Value::as_object) {
        for (mistake_name, mistake_val) in mistakes {
            let mut content = format!("Common Mistake - {}\n", titlecase(mistake_name));
            append_str(&mut content, mistake_val, "description");
            append_str(&mut content, mistake_val, "why_its_bad");
            append_str(&mut content, mistake_val, "instead");
            chunks.push(MiKnowledge {
                id: format!("mistake_{mistake_name}"),
                category: "common_mistakes".into(),
                topic: mistake_name.clone(),
                content,
            });
        }
    }

    tracing::info!(chunks = chunks.len(), "Loaded MI knowledge chunks");
    Ok(chunks)
}

/// Converts "snake_case" to "Title Case".
fn titlecase(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().to_string() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Appends a string field from a JSON object if it exists.
fn append_str(out: &mut String, obj: &Value, key: &str) {
    if let Some(s) = obj.get(key).and_then(Value::as_str) {
        out.push_str(s);
        out.push('\n');
    }
}

/// Appends an array-of-strings field from a JSON object as a labeled list.
fn append_str_array(out: &mut String, obj: &Value, key: &str, label: &str) {
    if let Some(arr) = obj.get(key).and_then(Value::as_array) {
        out.push_str(&format!("{label}: "));
        let items: Vec<&str> = arr.iter().filter_map(Value::as_str).collect();
        out.push_str(&items.join(". "));
        out.push('\n');
    } else if let Some(s) = obj.get(key).and_then(Value::as_str) {
        // Handle case where field is a string instead of array
        out.push_str(&format!("{label}: {s}\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_titlecase() {
        assert_eq!(titlecase("snake_case"), "Snake Case");
        assert_eq!(titlecase("open_questions"), "Open Questions");
        assert_eq!(titlecase("mi_spirit"), "Mi Spirit");
    }

    #[test]
    fn test_load_mi_principles() {
        let kb_path = PathBuf::from(
            std::env::var("PLOTINUS_KB_PATH")
                .unwrap_or_else(|_| {
                    "/home/sbeck/code/python/plotinus/data/knowledge_base".to_string()
                }),
        );
        if !kb_path.join("mi_principles.json").exists() {
            eprintln!("Skipping test: mi_principles.json not found at {}", kb_path.display());
            return;
        }

        let chunks = load_mi_principles(&kb_path).unwrap();

        // Should produce a reasonable number of chunks
        assert!(chunks.len() >= 20, "Expected at least 20 chunks, got {}", chunks.len());

        // Check categories are populated
        let categories: Vec<&str> = chunks.iter().map(|c| c.category.as_str()).collect();
        assert!(categories.contains(&"mi_spirit"));
        assert!(categories.contains(&"oars"));
        assert!(categories.contains(&"change_talk"));
        assert!(categories.contains(&"crisis"));

        // Each chunk should have non-empty content
        for chunk in &chunks {
            assert!(!chunk.content.is_empty(), "Chunk {} has empty content", chunk.id);
            assert!(!chunk.id.is_empty());
        }
    }
}
