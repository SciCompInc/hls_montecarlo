#  Equity Linked Note option pricing example

Equity Linked Notes (ELNs) allow clients to customize return to suit their
investment needs. Traditional equity investments provide full exposure to
the market, whether the performance is positive or negative. ELNs provide
clients with an alternative to traditional equities that can offer an
enhanced return if the underlying equity asset rises, and varying levels
of protection if the market falls. ELNs can be linked to a variety of 
underlying assets, including indices, single stocks, portfolios of shares
and industry sectors, and can be expanded to include commodities and 
currencies.

The mcSVEqLinkStructNote example prices an equity-linked structured note
in which the note is linked to a basket of indices. If on observation dates
the performance of all indices is above the knockout barrier for that date, 
the note redeems early at par plus a bonus coupon. If early
redemption does not occur, then at maturity either:

1. the Note redeems at par plus a maturity coupon, or
2. if the performance of at least `KINum` of the indices have even been below
   the knock in barrier during the tenor, on a continuously observed basis,
   then the Note redeems at the performance percentage of the worst index.

The indices follow correlated [Heston](https://en.wikipedia.org/wiki/Heston_model)
stochastic volatility processes and we allow for a term structure of rates.

There are two versions of the code:

* [Standard Monte Carlo](.) (this directory)

* [Quasi Monte Carlo](../mcSVEqLinkStructNoteQ)








