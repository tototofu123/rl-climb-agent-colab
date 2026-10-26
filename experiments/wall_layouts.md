# Wall Layouts

## Beginner (3 holds)

```
    (0)           (target)
  400,100         400,50
         
         
    (1)
  300,300
         
         
    (2)
  350,500
```

3 holds in a vertical line. Agent must go up sequentially. Max 2 moves.

## Intermediate (5 holds)

```
    (0)     (target)
  350,100   450,50
         
    (1)     (2)
  250,250   450,250
         
    (3)     (4)
  300,400   400,400
```

2 holds per row, target at top. Agent can choose path (left or right side).

## Advanced (7 holds + dyno required)

```
(target)
  400,50
         
    (0)         (1)
  250,150      550,150
         
    (2)  (3)   (4)
  200,300 400,300 600,300
         
    (5)         (6)
  300,450      500,450
```

Some holds are too far apart for static reach — agent must use dyno.

## Expert (10 holds, overhang)

```
(target)
  400,50
         
    (0)    (1)    (2)
  200,150 400,150 600,150
         
    (3)   (4)    (5)
  150,300 350,300 550,300
         
    (6)    (7)    (8)
  250,450 450,450 650,450
         
         (9)
        350,580
```

Wide holds, overhang section (holds get wider toward bottom), requires careful path planning and occasional dynos.

## Layout Format (YAML)

```yaml
name: beginner
width: 800
height: 600
holds:
  - pos: [400, 100]
    radius: 20
    type: jug
    is_target: false
  - pos: [300, 300]
    radius: 20
    type: jug
    is_target: false
  - pos: [350, 500]
    radius: 25
    type: jug
    is_target: false
  - pos: [400, 50]
    radius: 20
    type: jug
    is_target: true
```

## Curriculum Sequence

```
Stage 2 → Stage 3 → Stage 4/5 → Final
Beginner → Intermediate → Advanced (dyno unlocked) → Expert
```
