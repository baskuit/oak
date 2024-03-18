# Protocol

This project uses modified debug logs that store eval data

In each frame, in addition to the usual battle, choices, etc data, we have:

```
{log data}
{battle after commit}
{result}
{row choice}
{col choice}
{rows}
{cols}
row player's:
{
    {value}
    {row policy}
    {col policy}
    {n matrices}
    {matrices data}
}
col player's:
{
    {value}
    {row policy}
    {col policy}
    {n matrices}
    {matrices data}
}
```

Matrix Data consists of:

{rows}
{cols}
{n data elements}
{indivdual data type enums}
(
    0 = uint8_t
    1 = int32
    2 = float32
)
{
    matrix data in row - col - z order
}