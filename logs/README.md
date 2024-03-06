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
row players:
{
    {value}
    {row policy}
    {col policy}
    {n matrices}
    {matrices data}
}
col players:
{
    {value}
    {row policy}
    {col policy}
    {n matrices}
    {matrices data}
}
```