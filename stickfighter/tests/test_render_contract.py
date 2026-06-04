"""The renderer contract: reading a MatchSnapshot to draw a frame must never
mutate the sim state underneath it (render/ only ever reads)."""
