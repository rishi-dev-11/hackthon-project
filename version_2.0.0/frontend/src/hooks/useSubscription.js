import { useContext } from 'react';
import { SubscriptionContext } from '../context/SubscriptionContext';

export const useSubscription = () => useContext(SubscriptionContext);