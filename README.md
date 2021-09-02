# img2svg

My attempt to convert bitmap images into SVG vector images - very rubbish right now :)

## Usage

```
python main4.py
```

options:

`-i` specify input Image path. If not specified, it may choose a file in py file's directory.

`-o` specify output SVG path. If not specified, it will save in py file's directory.

`-q` estimated runtime in seconds. (or quality)

`-n_color` num of colors when grouping pixels. If not specified, It'll alter depending on quality. I recommand that 16<=`n_color`<=64.

`-nd` `-no_dots`, no dots.

`-nl` `-no_lines`, no lines.

